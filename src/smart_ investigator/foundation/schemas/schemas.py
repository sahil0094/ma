from __future__ import annotations
from typing import List, Optional, Annotated, Union, Any, Literal
from typing_extensions import TypedDict, NotRequired
from pydantic import BaseModel, Field, SkipValidation, ConfigDict, TypeAdapter, model_validator
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.serde.types import INTERRUPT
from smart_investigator.foundation.tools.tool_names import __all__ as tool_names
from enum import Enum
from dataclasses import import dataclass
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent
)
from mlflow.types.responses_helpers import ResponseOutputItemDoneEvent, OutputItem, Content
from mlflow.types.responses import ResponsesAgentStreamEvent, ResponseErrorEvent, ResponseTextDeltaEvent

class StandardToolArgument(BaseModel):
    text: str
    rationale: str

class ToolArgument(BaseModel):
    """TODO: change StateExchangeArguments to a generic LLM tool call if a different type of tool signature exist"""
    name: str = Field(description="Name of the tool to call. If empty, let the intent classifier decide.")
    arguments: str = Field(description="Arguments to the tool in name. Can be plain text or json string.")

class ToolStruct(BaseModel):
    """TODO: need to expand tool_argument for multi-tools call"""
    context: str = Field(description="Only used if tool_argument is None. Help the intent classifier know which tool to use and generate the corresponding arguments")
    # agent_output: FrontendOutput = Field(default={}, description="Objects related to the parent agent.")
    content: Content = Field(description="Content to be sent.")
    tool_argument: Optional[ToolArgument] = Field(description="Arguments to the tool if exist")

class ToolContentStruct(BaseModel):
    value: str
    interrupt: bool = Field(description="True means the task is not complete and requires assistance from other tools.")
    state: Optional[dict] = Field(description="The state of the task agent to return with.")
    next_tool: Optional[ToolStruct] = Field(description="Should only exist if interrupt = False.")
    trusted: bool = Field(default=False, description="If false, not using LLM intent classifier in the next_tool execution if it involved hitl.")
    artifact: dict[str, Any] = Field(default={}, description="Some agents can pass the artifact to other tasks. This mainly provide a result for the frontend.")
    error: bool = Field(default=False, description="Signal error from tool. Error message is stored in value")

class TaskStruct(BaseModel):
    parent_tool: str
    parent_state: Optional[dict | str]
    task_id: str
    task: ToolStruct

class ToolMetadata(BaseModel):
    expose_to_user: bool
    ic_introduction: Optional[str]
    can_generate_task: bool

class FrontendInputStruct(BaseModel):
    value: str # All input needs to be serialised
    trusted: bool = Field(description="True means no intent classifier required.")

class SmartInvestigatorAgentState(MessagesState):
    artifact: dict[str, Any]
    default_resume: bool

class FrontendInput(TypedDict):
    text: str
    artifact: dict
    is_direct: bool
    agent_name: Union[str, Literal[DECLINE_LETTER_AGENT_NAME, SMART_STRATEGY_AGENT_NAME]]

class UserInfo(TypedDict):
    user_name: str
    user_id: str
    user_upn: str

class WorkflowInfo(TypedDict):
    id: str
    stage: str
    mode: str

class ResponseAgentCustomInput(TypedDict):
    frontend_input: FrontendInput
    user_info: UserInfo
    time: str
    state: str
    is_resume: bool
    workflow: WorkflowInfo

class DatabricksOption(TypedDict):
    return_trace: bool

class SIResponseAgentRequest(BaseModel):
    user: str
    input: list[Message] = Field(default=[])
    custom_inputs: ResponseAgentCustomInput
    stream: bool
    databricks_options: DatabricksOption = Field(default=DatabricksOption(return_trace=True))

class EventType(str, Enum):
    # complete = "complete"
    # interrupt = INTERRUPT
    # success = "success"
    # Mlflow ResponsesAgentStreamEvent type https://github.com/mlflow/mlflow/blob/master/mlflow/types/responses.py#L70
    # Frontend event
    # Langgraph
    custom = "response.graph.custom"
    graph_update = "response.graph.updates"
    graph_done = "response.graph.done"
    # AzureOpenAI
    delta_text = "response.output_text.delta"
    done = "response.output_item.done"
    # Generic
    completed = "response.completed"
    error = "error"
    success = "success"

class FrontendArtifact(TypedDict):
    forms_to_render: NotRequired[SkipValidation[list[dict]]] = Field(default=None)
    metrics: NotRequired[SkipValidation[dict]] = Field(default=None)
    model_config = ConfigDict(extra="allow")

class FrontendOutput(TypedDict):
    index: Optional[int]
    # id: Optional[str]
    agent_name: Union[str, Literal[tuple(tool_names)]]
    text: str
    artifact: Optional[FrontendArtifact]

class ToolDict(TypedDict):
    name: str
    arguments: str

class MasterAgentContext(TypedDict):
    is_direct: bool
    context: str
    next_tool: Optional[ToolDict]

class SIErrorCode(str, Enum):
    LANGGRAPH_LOGIC = "LANGGRAPH_LOGIC"
    INTERFACE_MISMATCH = "INTERFACE_MISMATCH"
    AGENT_ENDPOINT_FAILURE = "AGENT_ENDPOINT_FAILURE"
    LLM_ENDPOINT_FAILURE = "LLM_ENDPOINT_FAILURE"
    UNKNOWN_FAILURE = "UNKNOWN_FAILURE"
    SUCCESS = "SUCCESS"

class ErrorStruct(TypedDict):
    error_description: str
    error_trace: str
    error_code: SIErrorCode

SUCCESS_ERROR_STRUCT = ErrorStruct(error_description="", error_trace="", error_code=SIErrorCode.SUCCESS)

class ResponseAgentCustomOutput(TypedDict):
    # frontend_outputs: FrontendOutput
    agent_name: Union[str, Literal[tuple(tool_names)]]
    artifact: Optional[FrontendArtifact]
    master_agent_context: Optional[MasterAgentContext] = Field(default=None)
    state: Union[dict, str]
    error: Optional[ErrorStruct]
    event_type: LanggraphStreamEvent

class SIResponsesAgentStreamEvent(ResponsesAgentStreamEvent):
    custom_outputs: ResponseAgentCustomOutput | dict

    # @model_validator(mode="after")
    # def check_type_si(self):
    #     type = self.type
    #     if type != EventType.graph_update:
    #         adapter = TypeAdapter(ResponseAgentCustomOutput)
    #         adapter.validate_python(self.custom_outputs)
    #     return self

    @model_validator(mode="after")
    def check_type_state(self):
        # pass
        # if not self.custom_outputs['state']:
        #     self.custom_outputs['state'] = {"haaha": "yeah"}
        return self

class LanggraphStreamEvent(str, Enum):
    updates = "updates"
    values = "values"
    custom = "custom"
    messages = "messages"
    thinking = "thinking"
    llm = "llm"
    done = "done"
    interrupt = INTERRUPT

class DoneType(str, Enum):
    thinking = "thinking"
    llm = "llm"
    response = "response"

class DoneCustomOutput(TypedDict):
    agent_name: str
    event: DoneType

class ContentCustomOutput(TypedDict):
    sender: str
    artifact: dict
    # forms_to_render: list[dict]
    # metrics: dict
    master_agent_context: MasterAgentContext

class ContentCustomInput(TypedDict):
    artifact: dict
    is_direct: bool
    agent_name: str
