from agents.orchestrator.master_agent_prompts import WELCOME_PROMPT, REPEAT_PROMPT, REJECT_FINISHED_WORKFLOW, REJECT_DIFFERENT_WORKFLOW
from agents.orchestrator.master_agent_utils import MasterAgentState, WorkflowStatus, IntentClassifierStruct, EndMessage, htil_handler, create_tool_calls_message, tool_parser, \
    get_default_context
from smart_investigator.foundation.tools.human_in_the_loop import human_in_the_loop
from smart_investigator.foundation.constants import MASTER_AGENT_NAME
from smart_investigator.foundation.schemas import SmartInvestigatorAgentState, SIResponseAgentStreamEvent, ToolStruct, TaskStruct, FrontEndInputStruct, StandardToolArgument, \
    ToolArgument, EventType, ResponseAgentCustomOutput, SIErrorCode, ContentCustomOutput
from smart_investigator.foundation.tools.common import get_hash_id, format_task_struct, EMPTY_HTIL_ARGUMENT, EMPTY_AGENT_ARGUMENT, get_contents
from tool_names import __all__ as tool_names
from typing import List, TypedDict, Optional, Annotated, Union, Any
from dataclasses import dataclass, field
from langgraph.core.graph import StateGraph, START, END
from langgraph.core.runnable import RunnableConfig, Runnable
from langgraph.core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langgraph.core.tools import ToolNode, InjectedState, tools_condition
from langgraph.prebuilt import ToolNode, InjectedState, tools_condition
from mlflow.types.responses_helpers import OutputItem, Content, Response
from copy import deepcopy
import json
import uuid
import mlflow


def _update_trace_context(tool_name: str, workflow: dict, thread_id: str = "") -> None:
    """
    Update MLflow trace with session context and tags for evaluation.

    Args:
        tool_name: Name of the tool being called
        workflow: Current workflow state dict
        thread_id: Session/thread identifier for grouping traces
    """
    try:
        mlflow.update_current_trace(
            metadata={
                "mlflow.trace.session": thread_id,  # Session grouping
            },
            tags={
                "tool_called": tool_name or "none",
                "is_hitl": str(tool_name == human_in_the_loop.name).lower(),
                "workflow_name": workflow.get("name", ""),
                "workflow_finished": str(workflow.get("is_finished", False)).lower(),
            }
        )
    except Exception:
        # Silently ignore if tracing is not active
        pass


def _traceback_direct_request(request_args: dict, task_stack: list[TaskStruct]) -> list[TaskStruct]:
    # Manage task stack in a direct request
    # custom_inputs = request_args.get("custom_inputs", {})
    contents = get_contents(request_args)
    content = next(get_contents(request_args))
    frontend_input = custom_inputs.get("frontend_input", {})
    is_direct = content.custom_inputs.get("is_direct", False)
    if is_direct:
        agent_name = content.custom_inputs.get("agent_name", False)
        if agent_name not in tool_names:
            raise Exception(f"[{MASTER_AGENT_NAME}] agent_name must be either {tool_names}. Got {agent_name}.", SIErrorCode.INTERFACE_MISMATCH)
        # return EndMessage(f"agent_name must be either {tool_names}. Got {agent_name}.")
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


def _extract_tool_name(ai_message: BaseMessage) -> str:
    # Extracting tool's name
    tool_calls = ai_message.tool_calls
    tool_name = ""
    if tool_calls:
        tool_name = tool_calls[0].get("name", "")
        if tool_name:
            print(f"**tool_name: {ai_message}")
    return tool_name


def _form_htil_tool_call(text: str):
    output_args = [
        Content(
            type="output_text",
            text=text,
            custom_outputs=ContentCustomOutput(
                sender=MASTER_AGENT_NAME,
            ),
        )
    ]

    return create_tool_calls_message(EMPTY_HTIL_ARGUMENT, human_in_the_loop.name, output_args)


def _process_tool_call_with_reject(ai_message: BaseMessage, workflow: dict, workflow_agents: list[str], non_workflow_agents: list[str]):
    tool_name = _extract_tool_name(ai_message)
    current_workflow_name = workflow.get("name", "")

    if tool_name not in workflow_agents:
        # Check if IC fell back to HITL AND workflow is finished
        if tool_name == human_in_the_loop.name and workflow.get("is_finished", False):
            # Replace clarification with rejection message
            reject_msg = _form_htil_tool_call(REJECT_FINISHED_WORKFLOW.format(
                workflow_name=current_workflow_name))
            return {"messages": [reject_msg]}
        # Otherwise proceed as usual (non-workflow agents work normally)
        return {"messages": [ai_message]}
    elif not workflow:
        # If there is no existing workflow, store and proceed as usual
        return {"messages": [ai_message], "workflow": WorkflowStatus(name=tool_name, is_finished=False)}
    elif current_workflow_name == tool_name:
        # Return to an existing workflow
        if workflow.get("is_finished", False):
            # Reject if the workflow is finished
            reject_msg = _form_htil_tool_call(REJECT_FINISHED_WORKFLOW.format(
                workflow_name=current_workflow_name))
            return {"messages": [reject_msg]}
        else:
            # Workflow not finished, proceed as usual
            return {"messages": [ai_message]}
    else:
        # Reject if a different workflow already started
        reject_msg = _form_htil_tool_call(REJECT_DIFFERENT_WORKFLOW.format(
            new_workflow_name=tool_name,
            current_workflow_name=current_workflow_name,
            list_workflow_string=", ".join([current_workflow_name] + non_workflow_agents)))
        return {"messages": [reject_msg]}


def get_graph(LM: BaseChatModel, tools: List[StructuredTool], max_last_messages: int = 1) -> StateGraph:
    # Form a string of tasks
    task_str = "\n- ".join([tool.name for tool in tools])
    workflow_agents = [tool.name for tool in list(filter(lambda x: x.metadata.is_workflow & (x.metadata.expose_to_user), tools))]
    non_workflow_agents = [tool.name for tool in list(filter(lambda x: (not x.metadata.is_workflow) & (x.metadata.expose_to_user), tools))]
    # print(f"workflow_agent {workflow_agents}")
    # print(f"non workflow agents {non_workflow_agents}")

    def intent_classifier(state: MasterAgentState) -> MasterAgentState:
        # print(f"**state: {state}")
        # Runtime = get_runtime()
        request_args = runtime.context.get("request", {})
        thread_id = getattr(request_args, "user", "") or ""  # Session identifier for trace grouping
        task_stack: List[Union[dict, TaskStruct]] = deepcopy(state.get("task_stack", []))  # will not change the input state when pop

        messages = state.get("messages", [])
        if not messages:
            # Start of a conversation
            # Form an introduction for WELCOME_PROMPT
            input_prompt = WELCOME_PROMPT.format(tasks=task_str)
            output_args = [
                Content(
                    type="output_text",
                    text=input_prompt,
                    custom_outputs=ContentCustomOutput(
                        sender=MASTER_AGENT_NAME,
                    ),
                )
            ]
            ai_message = create_tool_calls_message(EMPTY_HTIL_ARGUMENT, human_in_the_loop.name, output_args)
            task_args = json.dumps(dict(text=WELCOME_PROMPT.format(tasks=task_str)))
            # Update trace context for evaluation filtering (initial HITL welcome)
            _update_trace_context(human_in_the_loop.name, {}, thread_id)
            return {"messages": [ai_message], "task_stack": []}
        else:
            # Continuing conversation
            # Check the last tool message
            last_message = messages[-1]
            if isinstance(last_message, ToolMessage):
                # check if the tool is htil as we need to append HumanResponse to the messages
                tool_content: ResponseAgentCustomOutput = last_message.artifact
                # print_last_message = deepcopy(last_message)
                # print_last_message.state = "<state>" if last_message.state else "..."
                # print(f"**last_message: {print_last_message}")
                tool_content = [
                    Content(
                        type="output_text",
                        text=last_message.content,
                        custom_outputs=last_message.custom_outputs
                    )
                ]

                if last_message.status == EventType.error:
                    # Resend error from tool/agent to frontend (i.e. HITL)
                    # Not remove task
                    ares = json.dumps(dict(text=""))
                    ai_message = create_tool_calls_message(ares, human_in_the_loop.name, tool_content)
                    # return {"messages": [ai_message], "task_stack": task_stack}
                    raise Exception(f"Encountered error at agent {last_message.name}!", SIErrorCode.AGENT_ENDPOINT_FAILURE)

                # Should be ToolMessage
                # Task stack clean up
                if task_stack:
                    # Obtain the task of the most recent agent that interrupted
                    last_task: TaskStruct = format_task_struct(task_stack[-1])

                    if last_task.parent_tool == last_message.name:
                        # print("ACTION: Drop task_stack")
                        # last message coming from the last tool dealing the same task
                        task_stack.pop()  # drop the last task
                        # If the task was not complete, a new task with the same parent tool will be added
                        # This means same task & complete, drop task and move to the upper/parent one
                        # same task & incomplete, drop task and add a new task later

                if last_message.name == human_in_the_loop.name:
                    # HITL response only exists if MA has a checkpoint
                    # HITL operation will not add to the task_stack
                    human_message = HumanMessage(content=last_message.content, additional_kwargs=last_message.custom_outputs)
                    task_stack = _traceback_direct_request(request_args, task_stack)
                    last_messages = _get_last_human_messages(state.get("messages", []), max_last_messages)
                    ai_message = htil_handler(task_stack, state, LM, tools, request_args, last_messages=last_messages)
                    workflow = state.get("workflow", {})
                    return_state = _process_tool_call_with_reject(ai_message, workflow, workflow_agents, non_workflow_agents)
                    # Update trace context for evaluation filtering
                    tool_name = _extract_tool_name(ai_message)
                    _update_trace_context(tool_name, return_state.get("workflow", workflow), thread_id)
                    return {**return_state, "task_stack": task_stack}
                else:
                    # TODO: other special tools to handle? Check tool.metadata.can_generate_task? Handle node updates? OR text stream?
                    # check if this is a complete task
                    if not last_message.need_resume:
                        return_msg = {}
                        # Task is complete -> ignore next_tool in the tool_content
                        if task_stack:  # Tasks left
                            # Simply continue with the remaining task
                            last_task: TaskStruct = format_task_struct(task_stack[-1])
                            last_task_json = json.loads(last_task.tool_argument.arguments)  # TODO: should standardise???
                            last_task_content = deepcopy(last_task.content)
                            last_task_content.custom_outputs["master_agent_context"] = {}
                            last_task_content.custom_outputs["artifact"] = {}
                            output_args = deepcopy(tool_content)
                            # Append last_task_frontend_output and reset index
                            output_args.append(last_task_content)
                            output_args["frontend_outputs"] = frontend_outputs
                            ai_message = create_tool_calls_message(last_task.tool_argument.arguments, last_task.task.tool_argument.name, output_args)
                            return_msg = {"messages": [ai_message], "task_stack": task_stack}
                        else:
                            # No more task left -> either END the conversation or asking for a new task
                            if runtime.context.get("infinite_loop", True):  # Loop back
                                # Send prompt for the next task
                                arguments = EMPTY_HTIL_ARGUMENT
                                output_args = deepcopy(tool_content)
                                # append repeat prompt and reset index
                                output_args.append(Content(
                                    type="output_text",
                                    text=REPEAT_PROMPT.format(tasks=task_str),
                                    custom_outputs=ContentCustomOutput(
                                        sender=MASTER_AGENT_NAME,
                                    ),
                                ))
                                # output_args["frontend_outputs"] = frontend_outputs
                                ai_message = create_tool_calls_message(EMPTY_HTIL_ARGUMENT, human_in_the_loop.name, output_args)
                                # print(f"**ai_message: {ai_message}")
                                return_msg = {
                                    "messages": [ai_message],
                                    "task_stack": task_stack
                                }
                            else:
                                # End Master Agent
                                return_msg = {
                                    "messages": [EndMessage("", additional_kwargs=tool_content)],
                                    "task_stack": []
                                }

                        workflow = state.get("workflow", {})
                        current_workflow_name = workflow.get("name", "")
                        if current_workflow_name == last_message.name:
                            # Workflow is finished
                            workflow["is_finished"] = True
                            # Update workflow
                            return_msg = {**return_msg, "workflow": workflow}
                        return return_msg
                    else:
                        # Task is not complete
                        frontend_outputs = last_message.custom_outputs
                        custom_outputs = last_message.custom_outputs
                        master_agent_context = custom_outputs.get("master_agent_context", {})
                        context = master_agent_context.get("context", "")
                        if not context:
                            context = get_default_context(last_message.name, last_message.content)
                        next_tool = master_agent_context.get("next_tool", {})

                        # If MA does not define what to do next with the next_tool
                        if not next_tool:
                            next_tool = dict(
                                name=human_in_the_loop.name,
                                arguments=EMPTY_HTIL_ARGUMENT
                            )
                        tool_struct = ToolStruct(
                            context=context,
                            agent_output=frontend_outputs[0],
                            content=tool_content[0],
                            tool_argument=ToolArgument(**next_tool)
                        )

                        next_task = TaskStruct(
                            parent_tool=last_message.name,
                            parent_state=last_message.state,
                            task=tool_struct,
                            task_id=last_message.id
                        )
                        ai_message = create_tool_calls_message(tool_struct.tool_argument.arguments, tool_struct.tool_argument.name, tool_content)
                        return {
                            "messages": [ai_message],
                            "task_stack": task_stack + [next_task],
                        }

            elif isinstance(last_message, HumanMessage):
                # HITL's response without resuming will go here
                # Manage task stack in a direct request
                # custom_inputs = request_args.get("custom_inputs", {})
                content = get_first_content(request_args)
                # frontend_input = custom_inputs.get("frontend_input", {})
                is_direct = content.custom_inputs.get("is_direct", False)
                if is_direct:
                    agent_name = content.custom_inputs.get("agent_name", False)
                    if agent_name not in tool_names:
                        raise Exception(f"[{MASTER_AGENT_NAME}] agent_name must be either {tool_names}. Got {agent_name}.", SIErrorCode.INTERFACE_MISMATCH)
                    # return EndMessage(f"agent_name must be either {tool_names}. Got {agent_name}.")
                    for task in deepcopy(reversed(task_stack)):
                        parsed_task: TaskStruct = format_task_struct(task)
                        parent_tool = parsed_task.parent_tool
                        if parent_tool == agent_name:
                            # matching agent -> stop dropping
                            break
                        # keep dropping task until after getting the matching agent or no task left
                        task_stack.pop()
                human_message = HumanMessage(content=last_message.content, additional_kwargs=last_message.additional_kwargs)
                task_stack = _traceback_direct_request(request_args, task_stack)
                last_messages = _get_last_human_messages(state.get("messages", []), max_last_messages)
                ai_message = htil_handler(task_stack, state, LM, tools, request_args, last_messages=last_messages)
                workflow = state.get("workflow", {})
                return_state = _process_tool_call_with_reject(ai_message, workflow, workflow_agents, non_workflow_agents)
                # Update trace context for evaluation filtering
                tool_name = _extract_tool_name(ai_message)
                _update_trace_context(tool_name, return_state.get("workflow", workflow), thread_id)
                return {**return_state, "task_stack": task_stack}



            else:
                # TODO: Should not be here
                raise Exception(f"[{MASTER_AGENT_NAME}] Can't handle the type of the last message. Got {type(last_message)} with the content of:\n{last_message}", SIEErrorCode.LANGGRAPH_LOGIC)

        

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
