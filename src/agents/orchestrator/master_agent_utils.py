from agents.orchestrator.tools.control_test import agent_tools
from agents.orchestrator.agent_prompts import HILCC_PROMPT, ICC_PROMPT_TEMPLATE, REPEAT_PROMPT
from smart_investigator.foundation.tools.main_loop import MASTER_AGENT_NAME
from smart_investigator.foundation.tools.human_in_the_loop import human_in_the_loop, prepare_hitl_task
from dataclasses import dataclass
from typing import List, TypedDict, Optional, Annotated, Union, Any
from smart_investigator.foundation.schemas import (
    SmartInvestigatorAgentState,
    ToolStruct,
    TaskStruct,
    FrontendInputStruct,
    StandardToolArgument,
    ToolArgument,
    STResponseAgentStreamEvent,
    FrontendInput,
)
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode, InjectedState, tools_condition
from langgraph.runtime import get_runtime
from copy import deepcopy
import httpx
import json
import uuid
import backoff
import logging

logger = logging.getLogger(__name__)


class MasterAgentState(SmartInvestigatorAgentState):
    task_stack: List[TaskStruct]
    current_depth: int = 1  # TODO: control max_depth


class IntentClassifierStruct(BaseModel):
    tool: str = Field(description="Must be one of the exact name of the tool provided above.")
    text: str = Field(description="The input text into the tool.")
    rationale: str = Field(description="The question asks about coverage of a specific loss event.")


class EndMessage(AIMessage):
    pass


@dataclass
class MasterAgentContext:
    """Context for interrupt operations"""
    infinite_loop: bool = False


# Prepare objects
tool_parser = PydanticOutputParser(pydantic_object=STResponseAgentStreamEvent)
tools = [agent_tools, human_in_the_loop]


def get_default_context(agent_name: str, text: str) -> str:
    return f"{agent_name} was requesting for {text}"


def hitl_handler(
    state: MasterAgentState,
    llm: BaseChatModel,
    tools: List[StructuredTool],
    input_args: dict,
    output_args: dict,
    task_stack: List[TaskStruct],
) -> AIMessage:
    """Handle a HITL message (either directly from START or from hitl tool)."""
    content = get_first_content(input_args)
    custom_inputs = input_args.get("custom_inputs", {})
    frontend_input_struct = custom_inputs.get("frontend_input", {})
    try:
        frontend_input = FrontendInput(**frontend_input_struct)
    except Exception as e:
        raise Exception(
            f"[{MASTER_AGENT_NAME}] frontend_input should follow FrontendInput. Got {frontend_input_struct!r}."
        ) from e

    user_query = content.text
    is_direct = content.custom_inputs.get("is_direct", False)
    target_tool_name = content.custom_inputs.get("agent_name", "")

    if not is_direct:
        # Allow using llm for intent classifier
        if task_stack:
            last_task: TaskStruct = format_task_struct(task_stack[-1])
            context = last_task.context
        else:
            context = ""

        joined_last_messages = " ".join(["\n- " + msg for msg in last_messages])
        prompt_last_messages = (
            "Just for preference (i.e., the decision should be based primarily on the current user_query), "
            "the user also previously mentioned:\n<previous>\n"
            f"{joined_last_messages}\n</previous>"
            if joined_last_messages
            else ""
        )
        intro = (
            "You are an AI assistant specializing in routing insurance-related queries. "
            "Your task is to determine whether the user's query falls into one of provided tools."
        )
        single_tool_reminder = (
            "Following the description of each tool to determine which one the call. "
            "You must return one single tool call only."
        )

        if not context:
            context = ""

        IC_PROMPT_TEMPLATE = f"""{intro}

First, review the following context:
<context> {{{{context}}}} </context>.

Now, consider the user's current query:
<user_query> {{{{user_query}}}} </user_query>.

{{{{prompt_last_messages}}}}

The logic to determine the tool in the following order:
1) If the user_query is out of scope, call {human_in_the_loop.name} to decline an answer. "Always check if it is out of scope first as this is a strict condition."
2) If the context exists, prioritise calling the tool mentioned in the context if its description matches the intention of user_query and the user_query fully answer the context.
   * Must prioritise the tool in context before other tools unless it is out of scope.
3) If the user_query and the context matches partially, call {human_in_the_loop.name} for follow-up clarification.
4) If other tools than the one in the context matches the intention of user_query, call that tool.

{single_tool_reminder}
"""

        ic_prompt = IC_PROMPT_TEMPLATE.format(
            context=context,
            user_query=user_query,
            prompt_last_messages=prompt_last_messages,
        )

        def backoff_hdlr(details):
            logger.error(
                "Backing off {wait:0.1f} seconds after {tries} tries "
                "calling function {target} with args {args} and kwargs "
                "{kwargs}".format(**details)
            )

        @backoff.on_exception(
            backoff.expo,
            (httpx.RequestError, httpx.HTTPStatusError),
            giveup=lambda e: not is_retryable_exception(e),
            max_tries=3,
            jitter=backoff.full_jitter,
            on_backoff=backoff_hdlr,
        )
        def invoke_with_retry(tools: List[StructuredTool], prompt: str):
            return llm.bind_tools(tools=tools).invoke(prompt)

        logger.error(f"***Prompt: \n{ic_prompt}\n")
        ic_ai_message = invoke_with_retry(tools=tools, prompt=ic_prompt)

        if tools_condition(state["messages"] + [ic_ai_message]) == "END":
            # LLM message is not a tool calling
            add the content to hitl
            retry_prompt = (
                "The assistance is confused with the next step. (i.e. llm_message.content). "
                "Would you please retype a clearer response?"
            )
            content = json.dumps(dict(text=retry_prompt))
            ai_message = create_tool_calls_message(next_task, human_in_the_loop.name, content)
        else:
            ai_message = ic_ai_message

    else:
        # Trusted input
        # Llm classification is not used, rely only on task stack
        # Simply prevent the human content to the parent_tool
        if target_tool_name in [tool.name for tool in tools]:
            next_task = StandardToolArgument(text=user_query, rationale=f"A direct request for {target_tool_name}.").model_dump_json()
            ai_message = create_tool_calls_message(next_task, target_tool_name)
        else:
            # There should be no situation when blocking IC but no previous task to return
            # TODO: Should not be here
            raise Exception(f"[{MASTER_AGENT_NAME}] agent_name should not be empty.", SLErrorCode.INTERFACE_MISMATCH)

    return ai_message


def create_tool_calls_message(
    task_args: str,
    tool_name: str,
    output_args: dict[str, Any] | STResponseAgentStreamEvent | dict[str, Any] = {},
) -> AIMessage:
    """Generate an AIMessage with a single tool call.
    TODO: artifact to be changed to serialisable type???
    """
    tool_call = {
        "index": 0,
        "id": f"call_{uuid.uuid4()}",
        "function": {"arguments": task_args, "name": tool_name},  # TODO: check tool name and input struct
        "type": "function",
    }

    ai_message = AIMessage(
        content="",
        additional_kwargs={"tool_calls": [tool_call]},
    )

    if output_args:
        if isinstance(output_args, STResponseAgentStreamEvent):
            return inject_parameters(ai_message, output_args.model_dump())
        else:
            return inject_parameters(ai_message, output_args)

    return ai_message
