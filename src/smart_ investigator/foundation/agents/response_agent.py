from typing import Generator, Iterator, Any, List, TypedDict, Optional, Annotated, Union, NamedTuple
from uuid import uuid4
import mlflow

from langgraph.graph import StateGraph
from langgraph.errors import GraphInterrupt
from mlflow.models import set_model
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    _cc_stream_to_responses_stream,
)

from mlflow.types.responses_helpers import (
    OutputItem,
    Content,
    Response,
    ResponseOutputItemDoneEvent,
    ResponseErrorEvent,
    ResponseCompletedEvent,
    ResponseTextDeltaEvent,
)

from langchain_core.runnables import RunnableConfig
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
    BaseMessage,
    SystemMessage,
)
from langchain_core.messages.ai import AIMessageChunk

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.types import INTERRUPT
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver

from pydantic import BaseModel, Field

from smart_investigator.foundation.llm.poa_azure_openai import get_llm
# from smart_investigator.foundation.llm.naive_azure_openai import get_llm

from smart_investigator.foundation.schemas.schemas import (
    ToolContentStruct,
    EventType,
    DoneType,
    SIResponsesAgentStreamEvent,
    ResponseAgentCustomOutput,
    FrontendOutput,
    SIErrorCode,
    ErrorStruct,
    LanggraphStreamEvent,
    ContentCustomOutput,
)

from smart_investigator.foundation.utils.utils import handle_ai_message_chunk
from smart_investigator.foundation.utils.configs_utils import SecretNames
from smart_investigator.foundation.utils.nginx_utils import get_token
from smart_investigator.foundation.checkpointers.stateless_checkpointer.stateless_checkpointer import StatelessMemorySaver

from mlflow.pyfunc import PythonModelContext
from abc import abstractmethod, ABCMeta
from copy import deepcopy
import os
import yaml
from pathlib import Path
from dataclasses import dataclass
import logging
import uuid
import traceback

from databricks.sdk import WorkspaceClient
from more_itertools import peekable

logger = logging.getLogger(__name__)


# =========================
# Constants / Data Models
# =========================

@dataclass
class ReturnState:
    COMPLETE = "complete"
    INTERRUPT = INTERRUPT
    ERROR = "error"
    CUSTOM = "custom"
    DONE = ""


class CheckpointerConfig(BaseModel):
    checkpoint_obj: Optional[BaseCheckpointSaver]  # TODO: replace with just plain configuration

    class Config:
        arbitrary_types_allowed = True


# =========================
# Base Agent Definition
# =========================

class LanggraphResponsesAgent(ResponsesAgent, metaclass=ABCMeta):

    @property
    @abstractmethod
    def AgentName(self) -> str:
        pass

    @abstractmethod
    def get_graph(self) -> StateGraph:
        """
        Define how the langgraph's graph is obtained.
        Must return a graph.
        """
        pass


DEFAULT_ERROR_MESSAGE = (
    "The Orchestrator encountered an error and could not process the request. "
    "Please try again or escalate to L4 team."
)


# =========================
# Request Preprocessing
# =========================

def _preprocess_request(
    self,
    request: ResponsesAgentRequest,
    config: RunnableConfig,
    model_config: dict,
    checkpointer: Optional[BaseCheckpointSaver],
) -> dict:
    if not request.input:
        lg_input = {"messages": []}
    else:
        content = request.input[0].content[0]
        if isinstance(content, dict):
            content = Content(**content)

        text = content.text
        custom_inputs = content.custom_inputs

        human_message = HumanMessage(content=text, additional_kwargs={"custom_inputs": custom_inputs})

        if checkpointer:
            is_resume = request.custom_inputs.get("is_resume", False)
            history = checkpointer.get_tuple(config)

            if history and is_resume:
                if history.pending_writes:
                    lg_input = Command(resume=content, update={"default_resume": True})
                else:
                    lg_input = {"messages": [human_message]}
            else:
                lg_input = {"messages": [human_message]}
        else:
            lg_input = deepcopy(request.custom_inputs.get("state", {}))
            messages = deepcopy(lg_input.get("messages", []))
            messages.append(human_message)
            lg_input["messages"] = messages

    return lg_input


def feature_preprocess_request(
    self,
    request: ResponsesAgentRequest,
    config: RunnableConfig,
    model_config: dict,
    checkpointer: Optional[BaseCheckpointSaver],
    default_input: dict,
) -> dict:
    return default_input


# =========================
# Stream Event Builders
# =========================

def _prepare_done_response_agent_stream_event(self, input_dict: dict) -> list[Content]:
    try:
        messages = input_dict.get("messages", [])
        last_messages = input_dict["messages"][-1] if messages else None

        content = [
            Content(
                type="output_text",
                text=last_messages.content if last_messages else "",
                custom_outputs=ContentCustomOutput(
                    sender=self.model_config.get("model", {}).get("agent_name", {}),
                    artifact=input_dict.get("artifact", {}),
                    master_agent_context=input_dict.get("master_agent_context", {}),
                ),
            )
        ]
        return content
    except:
        return []


def _prepare_custom_response_agent_stream_event(self, input_dict: dict) -> list[Content]:
    try:
        content = [
            Content(
                type="output_text",
                text=input_dict.get("text", ""),
                custom_outputs=input_dict.get("custom_outputs", {}),
            )
        ]
        return content
    except:
        return []


def _prepare_llm_response_agent_stream_event(
    self,
    chunk: dict,
    metadata: dict,
    custom_outputs: dict
) -> Optional[ResponsesAgentStreamEvent]:
    if isinstance(chunk, AIMessageChunk):
        message, self.llm_content = handle_ai_message_chunk(
            chunk, self.llm_content, custom_outputs
        )
        return message
    else:
        pass


def _prepare_update_response_agent_stream_event(self, input_dict: dict) -> list[Content]:
    try:
        content = [
            Content(
                type="output_text",
                text="",
                custom_outputs=input_dict
            )
        ]
        return content
    except:
        return []


def _prepare_interrupt_response_agent_stream_event(self, content: list[Content]) -> list[Content]:
    try:
        return content
    except:
        return []


# =========================
# Feature Hooks
# =========================

def feature_prepare_done_response_agent_stream_event(
    self,
    input_dict: dict,
    default_content: list[Content]
) -> list[Content]:
    return default_content


def feature_prepare_custom_response_agent_stream_event(
    self,
    input_dict: dict,
    default_content: list[Content]
) -> list[Content]:
    return default_content


def feature_prepare_llm_response_agent_stream_event(
    self,
    chunk: dict,
    metadata: dict,
    default_event: Optional[SIResponsesAgentStreamEvent],
    original_llm_content: str,
    custom_outputs: dict
) -> Optional[SIResponsesAgentStreamEvent]:
    return default_event


def feature_prepare_update_response_agent_stream_event(
    self,
    input_dict: dict,
    default_content: list[Content]
) -> list[Content]:
    return default_content


def feature_prepare_interrupt_response_agent_stream_event(
    self,
    input_dict: Any,
    default_content: list[Content]
) -> list[Content]:
    return default_content


# =========================
# Post-processing
# =========================

def _postprocess_responses(self, mode: str, response: dict) -> SIResponsesAgentStreamEvent:
    if mode == LanggraphStreamEvent.updates:
        node, result = list(zip(response.keys(), response.values()))[0]
        content = self._prepare_update_response_agent_stream_event(result)
        feature_content = self.feature_prepare_update_response_agent_stream_event(result, content)
        pass

    elif mode == LanggraphStreamEvent.values:
        interrupt_payload = response.get(LanggraphStreamEvent.interrupt, None)

        if interrupt_payload:
            custom_outputs = interrupt_payload[-1].value
            content = self._prepare_interrupt_response_agent_stream_event(custom_outputs)
            feature_content = self.feature_prepare_interrupt_response_agent_stream_event(response, content)
            is_interrupt = True
        else:
            custom_outputs = response
            content = self._prepare_done_response_agent_stream_event(custom_outputs)
            feature_content = self.feature_prepare_done_response_agent_stream_event(response, content)
            is_interrupt = False

        yield ResponsesAgentStreamEvent(
            type=EventType.done,
            item=OutputItem(
                type="message",
                id=str(uuid4()),
                content=feature_content,
                custom_outputs=dict(event=DoneType.response, is_interrupt=is_interrupt)
            )
        )

    elif mode == LanggraphStreamEvent.messages:
        chunk, metadata = response
        custom_outputs = dict(sender=self.AgentName)
        original_llm_content = self.llm_content

        default_event = self._prepare_llm_response_agent_stream_event(chunk, metadata, custom_outputs)

        feature_event = self.feature_prepare_llm_response_agent_stream_event(
            chunk, metadata, default_event, original_llm_content, custom_outputs
        )

        if feature_event:
            yield feature_event

    elif mode == LanggraphStreamEvent.custom:
        stream_event = response.get("stream_event", None)
        if not stream_event:
            content = self._prepare_custom_response_agent_stream_event(response)
            feature_content = self.feature_prepare_custom_response_agent_stream_event(response, content)
            yield ResponsesAgentStreamEvent(
                type=EventType.done,
                item=OutputItem(
                    type="message",
                    id=str(uuid4()),
                    content=feature_content,
                    custom_outputs=dict(event=DoneType.thinking)
                )
            )
        else:
            yield stream_event


def feature_postprocess_responses(
    self,
    mode: str,
    response: dict,
    default_response: Iterator[SIResponsesAgentStreamEvent]
) -> SIResponsesAgentStreamEvent:
    yield from default_response


# =========================
# Error Builder
# =========================

def _create_simple_error_message(
    self,
    error_description: str = "",
    error_trace: str = "",
    error_code: str = ""
) -> ResponseAgentCustomOutput:
    return ResponseAgentCustomOutput(
        frontend_output=FrontendOutput(
            agent_name=self.AgentName,
            state={},
            text=DEFAULT_ERROR_MESSAGE,
            artifacts={},
        ),
        master_agent_context={},
        state={},
        error=ErrorStruct(
            error_description=error_description,
            error_trace=error_trace,
            error_code = error_code
    ),
    event_type = EventType.error
)

def load_context(self, context: PythonModelContext):
    artifact_path = context.artifacts.get('configs', '')
    if artifact_path:
        with open(artifact_path, "r", encoding="utf-8") as f:
            model_config = yaml.safe_load(f) or {}
    else:
        model_config = {}

    # nginx_configs = {
    #     SecretNames.WORKSPACE_URL: model_config.get("databricks", {}).get('workspace_url', ''),
    #     SecretNames.OPENAI_API_KEY: os.getenv(SecretNames.OPENAI_API_KEY),
    #     SecretNames.CLIENT_ID: os.getenv(SecretNames.CLIENT_ID),
    #     SecretNames.CLIENT_SECRET: os.getenv(SecretNames.CLIENT_SECRET),
    # }
    # os.environ[SecretNames.DATABRICKS_HOST] = nginx_configs[SecretNames.WORKSPACE_URL]
    # os.environ[SecretNames.DATABRICKS_TOKEN] = get_token(
    #     openai_key = os.getenv(SecretNames.OPENAI_API_KEY),
    #     client_id = os.getenv(SecretNames.CLIENT_ID),
    #     client_secret = os.getenv(SecretNames.CLIENT_SECRET),
    #     workspace_url = model_config.get("databricks", {}).get('workspace_url', '')
    # )

    self.llm: BaseChatModel = get_llm(model_config)
    # self.llm: Optional[BaseChatModel] = get_llm(os.getenv(SecretNames.OPENAI_API_KEY)) if nginx_configs[SecretNames.WORKSPACE_URL] and model_config else None
    # apikey = os.getenv("OPENAIKEY")
    # self.llm: Optional[BaseChatModel] = get_llm(apikey)
    self.graph = self.get_graph()

    # Prepare checkpointer config
    self.use_checkpointer = model_config['model']['use_checkpointer'] if model_config else False
    self.max_checkpoints = model_config['model']['max_checkpoints'] if model_config else 1

    # Prepare context
    context = model_config.get("model", {}).get("context", {})
    self.context = context if context else {}
    self.context["client"] = WorkspaceClient().serving_endpoints.get_open_ai_client()
    self.model_config = model_config
    self.llm_content = ''

def _init_checkpointer(self, request: ResponsesAgentRequest) -> Optional[BaseCheckpointSaver]:
    state_json = request.custom_inputs["state"]
    checkpointer = None if not self.use_checkpointer else StatelessMemorySaver(initial_state_json=state_json, max_checkpoints=self.max_checkpoints)
    self.checkpointer_config = CheckpointerConfig(checkpoint_obj=checkpointer)
    checkpoint_obj = self.checkpointer_config.checkpoint_obj
    if checkpoint_obj:
        checkpointer = checkpoint_obj
    else:
        checkpointer = None
    return checkpointer

def _init_agent(self, checkpointer: Optional[BaseCheckpointSaver]) -> CompiledStateGraph:
    if checkpointer:
        agent = self.graph.compile(checkpointer)
    else:
        agent = self.graph.compile()
    return agent

# @backoff.on_exception(backoff.expo, openai.RateLimitError)
# @mlflow.trace(span_type=SpanType.LLM) # TODO: track llm
def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    if request.custom_inputs:
        # Init graph operation: CompiledStateGraph, Checkpointer, input, and config
        config = {"configurable": {"thread_id": request.user}}
        checkpointer: Optional[BaseCheckpointSaver] = self._init_checkpointer(request)
        agent: CompiledStateGraph = self._init_agent(checkpointer)
        lg_input: dict = self._prepare_langgraph_input(request, config, checkpointer)
        try:
            response = agent.invoke(lg_input, config=config, context=self.context)
            interrupts = response.get(INTERRUPT, [])
            if interrupts:
                # TODO: Catch state return with interrupt
                response = interrupts[0].value.model_dump()
                output_item = self.create_text_output_item(text=ReturnState.INTERRUPT, id=interrupts[0].id)
                custom_outputs = self._prepare_interrupt_response_agent_stream_event(response)
                return ResponsesAgentResponse(output=[output_item], custom_outputs=response)
            else:
                # no interrupt, complete the task
                output_item = self.create_text_output_item(text=ReturnState.COMPLETE, id=str(uuid4()))
                custom_outputs = self._prepare_complete_response_agent_stream_event(response)
                return ResponsesAgentResponse(output=[output_item], custom_outputs=custom_outputs)
        except GraphInterrupt as e:
            response = e.args[0][0].value.model_dump()
            output_item = self.create_text_output_item(text=ReturnState.INTERRUPT, id=e.args[0][0].id)
            custom_outputs = self._prepare_interrupt_response_agent_stream_event(response)
            return ResponsesAgentResponse(output=[output_item], custom_outputs=response)
        except Exception as e:
            output_item = self.create_text_output_item(text=ReturnState.ERROR, id=str(uuid4()))
            if len(e.args)==0:
                ex_text = e.json()
                ex_code = SIErrorCode.UNKNOWN_FAILURE
            elif len(e.args)==1:
                ex_text = e.args[0]
                ex_code = SIErrorCode.UNKNOWN_FAILURE
            else:
                ex_text = e.args[0]
                ex_code = e.args[1]

            error_description = f"{self.AgentName} predict_stream error {type(e)}: {ex_text}"
            trace = traceback.format_exc()
            logger_str = f"{error_description}\n{trace}"
            logger.error(logger_str)
            custom_outputs = self._create_simple_error_message(error_description = error_description, error_trace = trace, error_code = ex_code)
            return ResponsesAgentResponse(output=[output_item], custom_outputs=custom_outputs)
    else:
        output_item = self.create_text_output_item(text=ReturnState.ERROR, id=str(uuid4()))
        custom_outputs = self._create_simple_error_message(error_description = "No custom_inputs")
        return ResponsesAgentResponse(output=[output_item], custom_outputs=custom_outputs)

def predict_stream(self, request: ResponsesAgentRequest | dict) -> Generator[ResponsesAgentResponse, None, None]:
    if request.custom_inputs:
        try:
            if isinstance(request, dict):
                request = ResponsesAgentRequest(**request)
            # print(f"request: {request}\n")
            # Init graph operation: CompiledStateGraph, Checkpointer, input, and config
            config = {"configurable": {"thread_id": request.user}}
            checkpointer: Optional[BaseCheckpointSaver] = self._init_checkpointer(request)
            agent: CompiledStateGraph = self._init_agent(checkpointer)
            default_input: dict = self._prepare_preprocess_request(request, config, self.model_config, checkpointer)
            lg_input: dict = self._feature_preprocess_request(request, config, self.model_config, checkpointer, default_input)
            final_response: Optional[SIResponsesAgentStreamEvent] = None

            # Combine initial context with request
            request_copy = deepcopy(request)
            request_copy.custom_inputs['state'] = {} # reduce payload as we don't need to keep state
            input_context = {**self.context, **{"request": request_copy.model_dump()}}
            # TODO: consider replace stream_mode="updates" to stream_mode="values" if the full state of the intermediate nodes are required (e.g. live time resetting)
            for mode, response in agent.stream(lg_input, stream_mode=self.model_config['model']['langgraph_event'], config=config, context=input_context):
                # print(f"***{mode} - {response}")
                try:
                    if mode!=LanggraphStreamEvent.updates:
                        default_response = self._postprocess_responses(mode, response) # <- determine the final output?
                        feature_response = self.feature_postprocess_responses(mode, response, default_response)

                        if not isinstance(feature_response, Iterator):
                            raise Exception(f"feature_response must be the type of Iterator. Got {type(feature_response)}", SIErrorCode.INTERFACE_MISMATCH)

                        peek_feature_response = peekable(feature_response)
                        # if not isinstance(peek_feature_response.peek(), SIResponsesAgentStreamEvent):
                        #    raise Exception(f"feature_response's element must be the type of SIResponsesAgentStreamEvent. Got {type(peek_feature_response.peek())}", SIErrorCode.INTERFACE_MISMATCH)

                        # Do not stream intermediate values
                        # The final values will be stream as a "done" event
                        if mode!=LanggraphStreamEvent.values:
                            yield from feature_response
                        else:
                            final_response = feature_response
                except Exception as e:
                    # TODO: any exceptions that we can ignore? Reraise all for now
                    raise e
        except GraphInterrupt as e:
            response = e.args[0] #e.args[0][0].value
            default_response = self._postprocess_responses(ReturnState.INTERRUPT, {LanggraphEvent.interrupt: response})
            final_response = self.feature_postprocess_responses('updates', response, default_response)
            yield final_response
        except Exception as e:
            # print(f"***{e}: {e.args}")
            if len(e.args)==0:
                ex_text = e.json()
                ex_code = SIErrorCode.UNKNOWN_FAILURE
            elif len(e.args)==1:
                ex_text = e.args[0]
                ex_code = SIErrorCode.UNKNOWN_FAILURE
            else:
                ex_text = e.args[0]
                ex_code = e.args[1]

            error_description = f"{self.AgentName} predict_stream error {type(e)}: {ex_text}"
            trace = traceback.format_exc()
            logger_str = f"{error_description}\n\n#### Trace ####\n{trace}"
            logger.error(logger_str)
            yield ResponsesAgentStreamEvent(type=EventType.error, code=ex_code, message=logger_str)

        # Finalise with an "response.done" message + a full payload.
        # Note that the last event from Langgraph or the interrupt will contain the full state and payload
        if final_response:
            # OutputItem type must be according to https://github.com/mlflow/mlflow/blob/master/mlflow/types/responses_helpers.py#L158
            final_response_event = next(final_response)
            yield ResponsesAgentStreamEvent(type=EventType.done, item=final_response_event.item)

            # TODO: conclude the stream with a finish event with item = OutputItem(type="computer_call") + and done delta_text stream?
            if checkpointer!=None:
                state = checkpointer.to_json()
            else:
                state = ''
            yield ResponsesAgentStreamEvent(
                type=EventType.completed,
                response=Response(output=[], metadata={'state': state}),
                status="completed",
            )
    else:
        custom_outputs = self._create_simple_error_message(error_description = "No custom_inputs")
        yield ResponsesAgentStreamEvent(type=EventType.error, message="No custom_inputs")
