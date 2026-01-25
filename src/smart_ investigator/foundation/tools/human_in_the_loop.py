from smart_investigator.foundation.utils.utils import tool_with_metadata, EMPTY_HITL_ARGUMENT, EMPTY_AGENT_ARGUMENT
from langgraph.prebuilt import ToolNode, InjectedState, tools_condition
from pydantic import BaseModel, Field
from langgraph.types import Command, interrupt
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage, SystemMessage
from langchain.tools import InjectedToolArg, InjectedToolCallId, ToolRuntime
from typing import List, TypedDict, Optional, Annotated, Union, Any
from smart_investigator.foundation.schemas.schemas import (
    ToolMetadata, ToolContentStruct, ToolStruct, ToolArgument, FrontendInputStruct, 
    LanggraphStreamEvent, SIResponsesAgentStreamEvent, ResponseAgentCustomOutput, 
    FrontendOutput, MasterAgentContext, ToolDict, FrontendArtifact, SIErrorCode, 
    ErrorStruct, ContentCustomOutput, EventType
)
from smart_investigator.foundation.tools.tool_names import MASTER_AGENT_NAME
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent
)
from mlflow.types.responses_helpers import OutputItem, Content, Response
from copy import deepcopy
import json
import uuid

# --- Constants --- [file:10]
OUT_OF_SCOPE_DESCRIPTION = """
Out of scope includes queries relating to the following topics: legal liability (liability to pay compensation for death or bodily injury to other people or loss or damage to their
property resulting from an incident), embezzlement of funds (excluding claimable events such as loss of rent from a tenant), office bearer's liability, voluntary workers cover,
separation, divorce, estrangement, domestic violence, abuse, death, financial hardship/distress (excluding claimable events such as loss of rent from a tenant), age, disability, mental
health, cognitive impairment, physical of a human being (excluding pets/animals), elder abuse, scams, financial difficulty, literacy or numeracy barriers, cultural and
linguistic diversity, Aboriginal or Torres Strait customers, remote locations, grief, gender, modern slavery, payments relating to the policy, refunds, premiums, sum insured,
recommendations about product/s, car policies or car claims, and unrealistic or fictional situations.
**Examples of queries that are classified as Out of scope queries:**
    - 'A guest fell and injured their head, am I covered?' - this relates to legal liability
    - 'My customers have separated, who do I pay?' - this relates to separation/divorce
    - 'Am I eligible for a full refund on my policy?' - this relates to payments relating to the policy
    - 'My claimant has passed, are they still covered for damages?' - this relates to death
"""

IRRELEVANT_TOOL_RESPONSE = """
Please consider rephrasing your question or providing more information.
"""

_AGENT_NAME = "human_in_the_loop"

# --- Human In The Loop Tool --- [file:9][file:8][file:7]
@tool_with_metadata(
    description=f"The tool that connect to human for input. Only call this tool if <user_query> and <context> does not make sense, related to unsupported tool, or require the follow up clarification.",
    metadata=ToolMetadata(expose_to_user=False, ic_introduction=None, can_generate_task=False),
    name_or_callable=_AGENT_NAME
)
def human_in_the_loop(
    text: Annotated[str, f"The input text should either a declination of answer or a follow up clarification question. Also include a short and concise rationale for your answer and list the available tools (do not list {_AGENT_NAME} as it is an internal operation)."],
    state: Annotated[dict, InjectedState],
    runtime: ToolRuntime,
    tool_call_id: Annotated[str, InjectedToolCallId],
    output_args: Annotated[list[Content], InjectedToolArg] = []
) -> ToolContentStruct:
    """state is the maintaining state of tool, not of the current (i.e. master agent)"""
    # Communicate with MasterAgent orchestration layer through ToolContentStruct
    if not output_args:
        if not text:
            raise Exception("HITL without an input text must have a non-empty output_args!", SIErrorCode.LANGGRAPH_LOGIC)
        
        # Master Agent asking for clarification or denying an answer
        # interrupt_obj = ResponseAgentCustomOutput(
        #     frontend_outputs = [FrontendOutput(
        #         index = 0,
        #         id = str(uuid.uuid1()),
        #         agent_name = agent_name,
        #         text = text,
        #         artifact = FrontendArtifact(
        #             forms_to_render = artifact.get("forms_to_render", []),
        #             metrics = artifact.get("metrics", {})
        #         )
        #     )],
        # )
        
        interrupt_obj = [
            Content(
                type='output_text',
                text=IRRELEVANT_TOOL_RESPONSE+text,
                custom_outputs=ContentCustomOutput(
                    sender=MASTER_AGENT_NAME,
                    forms_to_render={},
                    metrics={},
                    master_agent_context={}
                )
            )
        ]
    else:
        # TODO: validate output_args on type = EventType.interrupt and other things?
        interrupt_obj = deepcopy(output_args)
        # interrupt_obj['state'] = state

    # print(f"***interrupt_obj: {interrupt_obj}")
    content = interrupt(interrupt_obj) # Resume maintaining next_task structure
    
    # If MasterAgent is not equipped with a checkpointer, will never reach the below
    # print(f"***interrupt content: {content}")
    
    # return ToolMessage(
    #     content="",
    #     artifact=frontend_input_struct,
    #     tool_call_id=tool_call_id
    # )

    return ToolMessage(
        content=content.text,
        custom_outputs=content.custom_inputs,
        tool_call_id=tool_call_id,
        name=_AGENT_NAME,
        id=str(uuid.uuid1()),
        status=EventType.success,
        state={},
        need_resume=False
    )

# --- Helper Functions --- [file:7][file:6]
def prepare_hitl_task(agent_name: str, text: str, context: str, state: Optional[str]=None, is_direct: bool = True, artifact: dict[str,Any] = {}) -> list[Content]:
    arguments = json.dumps(dict(text = text))
    return [
        Content(
            type='output_text',
            text=text,
            custom_outputs=ContentCustomOutput(
                sender=agent_name,
                artifact=artifact,
                # forms_to_render=artifact.get("forms_to_render", []),
                # metrics=artifact.get("metrics", {}),
                master_agent_context=MasterAgentContext(
                    is_direct = True,
                    context = context if context else f"{agent_name} was requesting for `{text}`",
                    next_tool = ToolDict(
                        name = _AGENT_NAME,
                        arguments = arguments, #EMPTY_HITL_ARGUMENT,
                    )
                )
            )
        )
    ]
