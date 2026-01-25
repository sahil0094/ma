from agents.orchestrator.master_agent_prompts import WELCOME_PROMPT, TOOL_UPDATE, REPEAT_PROMPT
from agents.orchestrator.master_agent_utils import MasterAgentState, IntentClassifierStruct, EndMessage, hitl_handler, create_tool_calls_message, tool_parser, get_default_context
from smart_investigator.foundation.tools.tools_metadata import MASTER_AGENT_NAME
from smart_investigator.foundation.schemas import EventType, ResponseAgentCustomOutput, Content, SIEResponsesAgentStreamEvent, ToolStruct, TaskStruct, FrontEndInputStruct, StandardToolArgument, \
    ToolArgument, EventType, ResponseAgentCustomOutput, Content, SIEResponsesAgentStreamEvent, ToolStruct, TaskStruct, FrontEndInputStruct, StandardToolArgument, \
    ToolArgument, EventType, ResponseAgentCustomOutput, Content, SIEResponsesAgentStreamEvent, ToolStruct, TaskStruct, FrontEndInputStruct, StandardToolArgument, \
    ToolArgument, EventType, ResponseAgentCustomOutput, Content, SIEResponsesAgentStreamEvent, ToolStruct, TaskStruct, FrontEndInputStruct, StandardToolArgument
from smart_investigator.foundation.schemas import ContentCustomOutput
from smart_investigator.foundation.utils import get_hash_id, get_default_task_id
from dataclasses import dataclass
from typing import List, TypedDict, Optional, Annotated, Union, Any
from pydantic import BaseModel, Field
from langchain_core.outputs import PydanticOutputParser
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage, SystemMessage
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.core.tools import ToolNode, StructuredTool
from langgraph.runtime import Runtime, get_runtime
from langgraph.prebuilt import ToolNode, InjectedState, tools_condition
from mlflow.types.responses_helpers import OutputType, Content, Response
from copy import deepcopy
import json
import uuid


def _traceback_direct_request(request_args: dict, task_stack: list[TaskStruct]) -> list[TaskStruct]:
    # Manage task stack in a direct request
    custom_inputs = request_args.get("custom_inputs", {})
    content = get_first_content(request_args)
    frontend_input = custom_inputs.get("frontend_input", {})
    is_direct = content.custom_inputs.get("is_direct", False)
    if is_direct:
        agent_name = content.custom_inputs.get("agent_name", False)
        if agent_name not in tool_names:
            raise Exception(f"[{MASTER_AGENT_NAME}] agent_name must be either {tool_names}. Got {agent_name}.", SIEErrorCode.INTERFACE_MISMATCH)
            return EndMessage(f"agent name must be either {tool_names}. Got {agent_name}.")
        for task in deepcopy(reversed(task_stack)):
            parsed_task: TaskStruct = format_task_struct(task)
            parent_tool = parsed_task.parent_tool
            if parent_tool == agent_name:
                # matching agent -> stop dropping
                break
            # keep dropping task until after getting the matching agent or no task left
            task_stack.pop()
    return task_stack


def _get_last_human_messages(messages: list[BaseMessage], max_last_messages: int) -> list[str]:
    count = 0
    last_messages = []
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_messages.append(msg.content)
            count += 1
        if count >= max_last_messages:
            break
    return last_messages


def get_graph(llm: BaseChatModel, tools: List[StructuredTool], max_last_messages: int = 1) -> StateGraph:
    # Form a string of tasks
    task_str = "\n".join(f"- {tool.name}: {tool.metadata.ic_introduction}" for tool in list(filter(lambda x: x.metadata.expose_to_user, tools)))

    def intent_classifier(state: MasterAgentState) -> MasterAgentState:
        # print(f"###state: {state}")
        runtime: Runtime = get_runtime()
        request_args = runtime.context.get("request", {})
        task_stack: List[Union[dict, TaskStruct]] = deepcopy(state.get("task_stack", []))  # will not change the input state when pop

        messages = state.get("messages", [])
        if not messages:
            # Start of a conversation
            # Form an introduction for WELCOME_PROMPT
            input_prompt = WELCOME_PROMPT.format(tasks=task_str)
            # output_args = ResponseAgentCustomOutput(
            #     frontend_outputs = [FrontendOutput(
            #         index = 0,
            #         id = str(uuid.uuid4()),
            #         agent_name = MASTER_AGENT_NAME,
            #         text = input_prompt,
            #         artifact = None
            #     )],
            #     master_agent_context = None,
            #     state = state,
            #     error = None,
            #     event_type = EventType.interrupt
            # )
            output_args = [
                Content(
                    type="output_text",
                    text=input_prompt,
                    custom_outputs=ContentCustomOutput(
                        sender=MASTER_AGENT_NAME,
                    )
                )
            ]
            ai_message = create_tool_calls_message(EMPTY_HITL_ARGUMENT, human_in_the_loop.name, output_args)
            # task_args = json.dumps(dict(context=WELCOME_PROMPT.format(tasks=task_str)))
            return {"messages": [ai_message], "task_stack": []}

        else:
            # Continuing conversation
            # Check the last tool message
            last_message = messages[-1]
            if isinstance(last_message, ToolMessage):
                # check if the tool is hitl as we need to append Human Response to the messages
                # tool_content: ResponseAgentCustomOutput = last_message.artifact
                # print_last_message = deepcopy(last_message)
                # print_last_message.state = "<state>" if last_message.state else ""
                # print(f"###{last_message: {print_last_message}}")
                # print(f"###{last_message: {last_message}}")
                tool_content = [
                    Content(
                        type="output_text",
                        text=last_message.content,
                        custom_outputs=last_message.custom_outputs
                    )
                ]

                if last_message.status == EventType.error:
                    # Reraise error from tool/agent to frontend (i.e. HITL)
                    # Not remove task
                    args = json.dumps(dict(text=""))
                    ai_message = create_tool_calls_message(args, human_in_the_loop.name, tool_content)
                    return {"messages": [ai_message], "task_stack": task_stack}

                # Should be ToolMessage
                # Task stack clean up
                if task_stack:
                    last_task: TaskStruct = format_task_struct(task_stack[-1])

                    if last_task.parent_tool == last_message.name:
                        # print("ACTION: Drop task_stack")
                        # last message coming from the last tool demanding the same task
                        task_stack.pop()  # Drop the last task
                        # If the task was not complete, a new task with the same parent tool will be added
                        # This means same task & complete, drop and add to the upper/parent one
                        # same task & incomplete, drop task and add a new task later

                if last_message.name == human_in_the_loop.name:
                    # HITL response only exists if MA has a checkpoint
                    # HITL operation will not add to the task_stack
                    human_message = HumanMessage(content=last_message.content, additional_kwargs=last_message.custom_outputs)
                    task_stack = _traceback_direct_request(request_args, task_stack)
                    last_messages = _get_last_human_messages(state.get("messages", []), max_last_messages)
                    ai_message = hitl_handler(task_stack, llm, tools, request_args, last_messages=last_messages)
                    last_message = hitl_handler(task_stack, llm, tools, request_args, last_messages=last_messages)
                    return {"messages": [human_message, ai_message], "task_stack": task_stack}

                else:
                    # TODO: other special tools to handle? Check tool.metadata.can_generate_task? Handle node updates? OR text stream?
                    # Check if this is a complete task:
                    if not last_message.need_resume:
                        # Task is complete -> ignore next_tool in the tool_content
                        if task_stack:
                            # Tasks left
                            # Simply continue with the remaining task
                            last_task: TaskStruct = format_task_struct(task_stack[-1])
                            last_task_json = json.loads(last_task.task) if isinstance(last_task.task, str) else last_task.task
                            last_task_content = last_task_json.get("content", last_task.tool_argument.arguments)  # TODO: should standardise???
                            output_args = deepcopy(tool_content)
                            # Append last_task_frontend_output index
                            frontend_outputs = []
                            # for index, item in enumerate(tool_content):
                            #     item["index"] = index
                            #     frontend_outputs.append(item)
                            # last_task_frontend_output["index"] = index + 1
                            # output_args.append(last_task_content)
                            # output_args["frontend_outputs"] = frontend_outputs
                            ai_message = create_tool_calls_message(last_task.task.tool_argument.arguments, last_task.task.tool_argument.name, output_args)
                            return {"messages": [ai_message], "task_stack": task_stack}

                        else:
                            # No more task left -> either END the conversation or asking for a new task
                            if runtime.context.get("infinite_loop", True):  # Loop back
                                # Send prompt for the next task
                                arguments = EMPTY_HITL_ARGUMENT
                                output_args = deepcopy(tool_content)
                                # Append repeat prompt and reset index
                                # frontend_outputs = []
                                # for index, item in enumerate(output_args["frontend_outputs"]):
                                #     item["index"] = index
                                #     frontend_outputs.append(item)
                                # output_args.append(Content(
                                #     type="output_text",
                                #     text=REPEAT_PROMPT.format(tasks=task_str),
                                #     custom_outputs=ContentCustomOutput(
                                #         sender=MASTER_AGENT_NAME,
                                #     )
                                # ))
                                output_args.append(Content(
                                    type="output_text",
                                    text=REPEAT_PROMPT.format(tasks=task_str),
                                    custom_outputs=ContentCustomOutput(
                                        sender=MASTER_AGENT_NAME,
                                    )
                                ))
                                # output_args["frontend_outputs"] = frontend_outputs
                                ai_message = create_tool_calls_message(EMPTY_HITL_ARGUMENT, human_in_the_loop.name, output_args)
                                # print(f"###{ai_message: {ai_message}}")
                                return {"messages": [ai_message], "task_stack": []}

                            else:
                                # End Master Agent
                                return {
                                    "messages": [EndMessage("", additional_kwargs=tool_content)],
                                    "task_stack": []
                                }

                    else:
                        # Task is not complete
                        frontend_outputs = last_message.custom_outputs
                        custom_outputs = last_message.custom_outputs
                        master_agent_context = custom_outputs.get("master_agent_context", {})
                        context = master_agent_context.get("context", {})
                        if not context:
                            context = get_default_context(last_message.name, last_message.content)
                        next_tool = master_agent_context.get("next_tool", {})
                        if not next_tool:
                            next_tool = {
                                "name": human_in_the_loop.name,
                                "arguments": EMPTY_HITL_ARGUMENT
                            }
                        # task_id = last_message.id
                        # if task_id:
                        #     task_id = last_message.id
                        tool_content = Content(
                            type="output_text",
                            text=last_message.content,
                            custom_outputs={}
                        )
                        tool_struct = ToolStruct(
                            context=context,
                            agent_output=frontend_outputs,
                            content=tool_content,
                            tool_argument=ToolArgument(**next_tool)
                        )
                        next_task = TaskStruct(
                            parent_tool=last_message.name,
                            parent_state=last_message.state,
                            task=tool_struct,
                            task_id=last_message.id
                        )
                        ai_message = create_tool_calls_message(tool_struct.tool_argument.arguments, tool_struct.tool_argument.name, [tool_content])
                        return {
                            "messages": [ai_message],
                            "task_stack": task_stack + [next_task]
                        }

            elif isinstance(last_message, HumanMessage):
                # HITL's response without resuming will go here
                pass

            else:
                # TODO: Should not be here
                raise Exception(f"[{MASTER_AGENT_NAME}] Can't handle the type of the last message. Got {type(last_message)} with the content of:\n{last_message}", SIEErrorCode.LANGGRAPH_LOGIC)

        return state

    def end_condition(state: MasterAgentState) -> bool:
        last_message = state.get("messages", [])[-1]
        if isinstance(last_message, EndMessage):
            # End MasterAgent if the last message is EndMessage
            tasks = state.get("task_stack", [])
            messages_length = len(state.get("messages", []))
            # End if there is no pending task and messages length is larger than 1 (not a init of the conversation)
            if not tasks and (messages_length > 1):
                return True
        return False

    graph = (
        StateGraph(MasterAgentState)
        .add_node("intent_classifier", intent_classifier)
        .add_node("tool_node", ToolNode(tools=tools))
        .add_edge(START, "intent_classifier")
        .add_edge("intent_classifier", "tool_node")
        .add_conditional_edges("intent_classifier", end_condition, {True: END, False: "tool_node"})
        .add_edge("tool_node", "intent_classifier")
    )
    return graph
