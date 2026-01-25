from langgraph.types import StreamWriter
from langgraph.checkpoint.serde.types import INTERRUPT
from langgraph.prebuilt import ToolNode, InjectedState, tools_condition
from langgraph.runtime import Runtime, get_runtime
from langgraph.config import get_stream_writer
from langchain.tools import InjectedToolArg, InjectedToolCallId, ToolRuntime
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage, SystemMessage
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent
)
from copy import deepcopy
from typing import List, Tuple, TypedDict, Optional, Annotated, Union, Any, Generator
import json
from fastapi.encoders import jsonable_encoder
from smart_investigator.foundation.agents.response_agent import LanggraphResponsesAgent
from smart_investigator.foundation.utils.utils import (
    tool_with_metadata, 
    format_task_struct, 
    get_last_message, 
    is_retryable_exception, 
    backoff_second, 
    prepare_thinking_message
)
from smart_investigator.foundation.utils.configs_utils import SecretNames
from smart_investigator.foundation.schemas.schemas import (
    ToolMetadata, 
    ToolContentStruct, 
    TaskStruct, 
    EventType, 
    ResponseAgentCustomOutput, 
    FrontendOutput, 
    SIErrorCode, 
    DoneType
)
from smart_investigator.foundation.tools.tool_names import MASTER_AGENT_NAME
from mlflow.deployments import get_deploy_client
from mlflow.types.responses_helpers import (
    OutputItem, 
    Content, 
    Response, 
    ResponseOutputItemDoneEvent, 
    ResponseErrorEvent, 
    ResponseCompletedEvent, 
    ResponseTextDeltaEvent
)
from databricks.sdk import WorkspaceClient
from enum import Enum
import os
import uuid
import httpx
from httpx_sse import connect_sse
import time
import logging

logger = logging.getLogger(__name__)

def _request_to_dict(request: ResponsesAgentRequest, stream: bool) -> dict:
    return jsonable_encoder(dict(
        user=request.user,
        input=request.input,
        custom_inputs=request.custom_inputs,
        stream=stream
    ))

class ToolFactory:
    def __init__(self, agent: Optional[LanggraphResponsesAgent], tool_config: dict):
        self.tool_config = tool_config
        self.client = WorkspaceClient().serving_endpoints.get_open_ai_client()
        self.agent = agent

    def _open_stream(self, request: Union[dict, ResponsesAgentRequest]) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """
        Return an iterator/stream of events from either:
        - self.agent.predict_stream (local agent)
        - Mosaic AI responses endpoint
        """
        if self.agent:
            # Databricks Agent streaming
            return self.agent.predict_stream(request)
        else:
            # Mosaic endpoint streaming
            # request here is assumed to be a dict
            return self.client.responses.create(
                model=self.tool_config['endpoint_name'],
                input=request.get("input", []),
                stream=True,
                extra_body={
                    "user": request.get("user", []),
                    "custom_inputs": request.get("custom_inputs", {}),
                    "databricks_options": dict(return_trace=True),
                },
            )

    def _get_final_event(self, request: Union[dict, ResponsesAgentRequest], writer: StreamWriter) -> Tuple[ResponsesAgentStreamEvent, dict]:
        """
        Stream with retry semantics:
        - Retry a few times if we get a *transient* error and haven't streamed anything yet.
        - If we've already emitted events to the caller, do NOT auto-retry: propagate error.
        """
        max_attempts = 3
        attempt = 0

        while attempt < max_attempts:
            attempt += 1
            received_any = False
            final_response: ResponsesAgentStreamEvent | None = None
            try:
                stream = self._open_stream(request)
                state = ""
                for event in stream:
                    received_any = True
                    if event.type == 'response.output_text.delta':
                        ml_event = ResponsesAgentStreamEvent(delta=event.delta, item_id=event.item_id, type=event.type)
                        llm_event = prepare_thinking_message(stream_event=ml_event)
                        writer(llm_event)
                    elif event.type == 'response.output_item.done':
                        custom_outputs = event.item.custom_outputs
                        if not custom_outputs:
                            raise Exception("Feature agent must return a non-empty custom_outputs!", SIErrorCode.INTERFACE_MISMATCH)
                        else:
                            event_type = custom_outputs.get("event", "")
                            if (event_type == DoneType.llm) or (event_type == DoneType.thinking):
                                ml_event = ResponsesAgentStreamEvent(
                                    item=OutputItem(
                                        type=event.item.type,
                                        id=event.item.id,
                                        custom_outputs=event.item.custom_outputs,
                                        content=[Content(
                                            type=content.type,
                                            text=content.text,
                                            custom_outputs=content.custom_outputs
                                        ) for content in event.item.content]
                                    ),
                                    type=event.type
                                )
                                llm_event = prepare_thinking_message(stream_event=ml_event)
                                writer(llm_event)
                            elif event_type == DoneType.response:
                                final_response = event
                            else:
                                raise Exception(f"custom_outputs['event'] must be either 'llm', 'thinking', 'response'. Got '{event_type}'.", SIErrorCode.INTERFACE_MISMATCH)
                    elif event.type == 'response.completed':
                        state = event.response.metadata.get('state', '')
                    else:
                        logger.error(f"Unsupported event! Got\n{event}")

                if final_response:
                    return final_response, state

                raise RuntimeError(f"[{self.tool_config['name']}] Stream completed without final event", SIErrorCode.AGENT_ENDPOINT_FAILURE)

            except Exception as exc:
                if not is_retryable_exception(exc) or received_any or attempt >= max_attempts:
                    raise Exception(f"[{self.tool_config['name']}] Error during streaming: {exc}", SIErrorCode.AGENT_ENDPOINT_FAILURE)

                sleep_seconds = backoff_second(attempt)
                logger.error(f"Retry streaming to {self.tool_config['name']}. Got {exc}")
                custom_message = prepare_thinking_message(
                    agent_name=self.tool_config['name'],
                    text=f"Retrying getting responses from {self.tool_config['name']} after {sleep_seconds}s."
                )
                writer(custom_message)
                time.sleep(sleep_seconds)
                continue

        raise RuntimeError(f"[{self.tool_config['name']}] Unexpected retry loop exit", SIErrorCode.AGENT_ENDPOINT_FAILURE)

    def create_tool(self):
        @tool_with_metadata(
            name_or_callable=self.tool_config['name'],
            description=self.tool_config['description'],
            metadata=ToolMetadata(
                expose_to_user=self.tool_config['metadata']['expose_to_user'],
                ic_introduction=self.tool_config['metadata']['ic_introduction'],
                can_generate_task=self.tool_config['metadata']['can_generate_task']
            ),
        )
        def tool_func(
            text: Annotated[str, self.tool_config['text_annotation']],
            rationale: Annotated[str, self.tool_config['rationale_annotation']],
            state: Annotated[dict, InjectedState],
            config: RunnableConfig,
            runtime: ToolRuntime,
            tool_call_id: Annotated[str, InjectedToolCallId],
            output_args: Annotated[dict, InjectedToolArg] = {}
        ) -> ToolContentStruct:
            writer: StreamWriter = get_stream_writer()
            custom_message = prepare_thinking_message(
                agent_name=MASTER_AGENT_NAME,
                text=f"Choosing {self.tool_config['name']} due to: {rationale}"
            )
            writer(custom_message)
            runtime_env: Runtime = get_runtime()
            request = deepcopy(runtime_env.context.get("request", {}))
            
            if not request:
                raise Exception(f"[{self.tool_config['name']}] empty request_args", SIErrorCode.LANGGRAPH_LOGIC)

            if not self.tool_config['checkpointer']:
                task_stack = state.get('task_stack', [])
                input_state = {}
                if task_stack:
                    last_task: TaskStruct = format_task_struct(task_stack[-1])
                    if last_task.parent_tool == self.tool_config['name']:
                        input_state = deepcopy(last_task.parent_state)
            else:
                input_state = {}

            request['custom_inputs']['state'] = input_state
            
            final_event, state_out = self._get_final_event(request, writer)
            contents = final_event.item.content
            if not contents:
                raise Exception('Langgraph event.item.content must not be empty!', SIErrorCode.INTERFACE_MISMATCH)
            
            content = contents[0]
            return ToolMessage(
                content=content.text,
                custom_outputs=content.custom_outputs,
                tool_call_id=tool_call_id,
                name=self.tool_config['name'],
                id=str(uuid.uuid1()),
                status=EventType.error if final_event.type == EventType.error else EventType.success,
                state=state_out,
                need_resume=final_event.item.custom_outputs.get('is_interrupt', False)
            )
        return tool_func
