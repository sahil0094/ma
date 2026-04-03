from typing import List, TypedDict, Optional, Annotated, Union, Any, Literal, Tuple, Dict
from smart_investigator.foundation.schemas.schemas import SmartInvestigatorAgentState, SIErrorCode
from smart_investigator.foundation.tools.tool_names import EXTERNAL_AGENT_NAME
from agents.external_agent.prompt_manager.knowledge_prompts import KNOWLEDGE_RETRIEVAL_SYSTEM_PROMPT, KNOWLEDGE_REPORT_SYSTEM_PROMPT, KNOWLEDGE_REPORT_PROMPT, RETRIEVAL_TASKS, KNOWLEDGE_RETRIEVAL_TASK_PROMPT
# from agents.external_agent.prompt_manager.interview_strategy_prompts import INTERVIEW_STRATEGY_SYSTEM_PROMPT, INTERVIEW_STRATEGY_DRAFT_PROMPT, INTERVIEW_STRATEGY_FEEDBACK_PROMPT
# from agents.external_agent.prompt_manager.external_agent_prompts import EXTERNAL_AGENT_SYSTEM_PROMPT, KEY_CONCERNS_DRAFT_PROMPT, INTERVIEW_DOC_REQUEST_PROMPT,
# INTERVIEW_PLAN_FEEDBACK_PROMPT, DOC_REQUEST_DRAFT_PROMPT, ADDITIONAL_ENQUIRIES_DRAFT_PROMPT, INTERVIEW_PLAN_DRAFT_PROMPT
# from agents.external_agent.prompt_manager.online_eval_prompts import eval_user_msg_template, eval_sys_msg_template
from agents.external_agent.tools.query_investigation_processes import query_investigation_processes
from agents.external_agent.tools.think_tool import think_tool
from agents.external_agent.tools.search_complete import search_complete
from agents.external_agent.utils import build_form_info, build_form_strategy, parse_form_to_interview_strategy, build_form_plan, parse_form_to_interview_plan, build_form_final
from smart_investigator.foundation.utils.utils import prepare_thinking_message, prepare_hitl_task
from agents.external_agent.schemas import InterviewStrategy, InterviewQuestion, InterviewQuestionSets, DocRequest, DocRequestSet, InterviewPlan, KnowledgeSet, KnowledgeReport, Knowledge, HITLDecision, InterviewPlanState, KeyConcern, KeyConcernSet, ExternalAgentPlan, AdditionalEnquiriesSet, AdditionalEnquiries

from smart_investigator.foundation.evals.online.eval_core import run_evaluation_from_yaml

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.config import get_stream_writer
from langgraph.types import Command, interrupt, StreamWriter
from langgraph.runtime import Runtime, get_runtime
from copy import deepcopy
import json
import logging
import uuid
from datetime import datetime
import asyncio
import mlflow
from mlflow.entities import SpanType

logger = logging.getLogger(__name__)

use_checkpointer = True  # TODO get from config


def get_graph(llm: BaseChatModel) -> StateGraph:
    # ------------------------------------
    # Helpers
    # ------------------------------------

    def _get_ctx(state: dict) -> tuple[Runtime, StreamWriter, list]:
        runtime = get_runtime()
        writer = get_stream_writer()
        messages = state.get("messages", []) or []
        return runtime, writer, messages

    def _frontend_input(runtime: Runtime) -> dict[str, Any]:
        req = runtime.context.get("request", {}) or {}

        return (req.get("input", {}) or {})[0].get("content", {})[0] or {}

    def _classify_hitl(
        hitl_text: str,
        hitl_artifact: dict[str, Any],
        prev_task: str
    ) -> HITLDecision:
        """
        Classification rules:
        - If artifact is present, and no text => accept (user edited/confirmed structured output)
        - Else if text is present => classify intent & summarise task using LLM
        """

        if hitl_artifact and not hitl_text:
            return HITLDecision(intent="accept", task_summary="")

        # if hitl_text.strip():
        #    # Treat plain text as feedback by default (this matches your stated behavior)
        #    return HITLDecision(intent="feedback", hitl_text.strip())

        parser = PydanticOutputParser(pydantic_object=HITLDecision)
        prompt = f"""
        You are routing and summarising the next task for a human-in-the-loop response for an insurance fraud interview plan workflow. The task should either be to draft an interview plan, or to edit based on user feedback.

        Previous task: {prev_task}

        Human response text:
        {hitl_text if hitl_text else "<empty>"}

        Human response artifact:
        {hitl_artifact if hitl_artifact else "<empty>"}

        Classify intent:
        - "accept": user accepts OR the artifact has been provided back with no human response text.
        - "feedback": user gives feedback/instructions to revise.
        - "unrelated": user asks something unrelated.

        Summarise the task:
        - If the intent is unrelated, or if there is no human response text, leave the task as ""
        - Otherwise, write a short paragraph that states the next task.
        - Start the task_summary with "Task: "

        {parser.get_format_instructions()}
        Return JSON only.
        """

        resp = llm.invoke(
            input=prompt,
            temperature=0.0,
            max_tokens=512,
            response_format={"type": "json_object"},
        )
        content = resp.content if isinstance(
            resp.content, str) else resp.content[0]["text"]
        hitl_decision = parser.parse(content)

        return hitl_decision

    def _route_from_pending_step(pending_step: str) -> str:
        """
        Where to go next after classifying the HITL.
        - unrelated always restarts
        - accept/feedback re-enters the relevant node
        """
        # accept/feedback go back to the generating node for that step
        if pending_step == "init":
            return "initialise_query"
        # if pending_step == "strategy_review":
        #     return "generate_strategy"
        if pending_step == "plan_review":
            return "generate_plan"

        # unknown step -> restart
        return "initialise_query"

    def _resolve_hitl_artifact(state: "InterviewPlanState", pending_step: str, incoming_artifact: dict | None) -> Any:
        """
        If the frontend didn't send an artifact (common for feedback-only resumes), fall back to the last generated output stored in state for the relevant step.
        """
        incoming_artifact = incoming_artifact or {}

        if pending_step == "strategy_review":
            # prev: InterviewStrategy | None = state.get("interview_strategy")
            # if incoming_artifact:
            #     payload = incoming_artifact
            #     if isinstance(payload, dict):
            #         return parse_form_to_interview_strategy(payload, previous=prev)

            # return prev
            pass

        if pending_step == "plan_review":
            prev: InterviewPlan | None = state.get("interview_plans")
            if incoming_artifact:
                payload = incoming_artifact
                if isinstance(payload, dict):
                    return parse_form_to_interview_plan(payload, previous=prev)
            return prev

        # For init/unknown, nothing sensible to fall back to
        return incoming_artifact


def _parse_investigation_type_to_filters(lob: str, inv_type: str) -> dict:
    """
    Input examples:
    - lob: "Motor", investigation_type: "Staged accident"
    - lob: "Motor", investigation_type: "Misrepresentation | sub-type"

    Output:
    {"lob": ["Motor"], "fraud_type": ["staged accident"]}
    """
    s = (inv_type or "").strip()

    # Fraud type is before the sub type delimiter "|"
    fraud_type = s.split("|", 1)[0].strip()

    if not fraud_type:
        raise ValueError(
            f"Could not parse fraud_type from investigation_type: {inv_type}")

    return {"lob": [lob], "fraud_type": [fraud_type]}

# ------------------------------------
# Nodes
# ------------------------------------


def route_interrupt(state: "InterviewPlanState") -> Command:
    """
    Central interrupt handler:
    1) Calls interrupt(...) exactly once per HITL prompt.
    2) On resume, classifies accept/feedback/unrelated.
    3) Extracts a concise task summary for the agent.
    4) Routes back to the correct node based on pending_step.
    """
    runtime, writer, messages = _get_ctx(state)

    pending_step = state.get("pending_step")
    hitl_task = state.get("hitl_task")
    prev_decision = state.get("hitl_decision")

    # - First pass: pauses execution here.
    # - Resume: returns immediately with resume payload (text).
    _ = interrupt(hitl_task)

    hitl = _frontend_input(runtime)
    hitl_text = (hitl.get("text") or "").strip()
    incoming_artifact = hitl.get("custom_inputs", {}).get(
        "artifact").get("form_data") or {}
    prev_task = prev_decision.task_summary if prev_decision else ""

    hitl_decision = _classify_hitl(hitl_text, incoming_artifact, prev_task)
    hitl_artifact = _resolve_hitl_artifact(
        state, pending_step, incoming_artifact)

    intent = hitl_decision.intent
    task_summary = hitl_decision.task_summary

    if intent == "unrelated":
        text = "Please review the AI generated output and either edit and submit, or provide your feedback."
        new_hitl_task = prepare_hitl_task(
            agent_name=EXTERNAL_AGENT_NAME,
            text=text,
            context="User must accept, edit+submit, or provide feedback relevant to the current output.",
            state={} if use_checkpointer else {
                **state, "messages": messages + [AIMessage(text)]},
            artifact=hitl_artifact,
        )

        writer(prepare_thinking_message(EXTERNAL_AGENT_NAME,
               "HITL response unrelated; re-prompting user."))
        return Command(
            goto="route_interrupt",
            update={
                "hitl_task": new_hitl_task,
                "hitl_decision": hitl_decision,
                "hitl_artifact": hitl_artifact,
                "messages": messages + [AIMessage(f"HITL decision: unrelated (step={pending_step})")],
            },
        )

    custom_message = prepare_thinking_message(
        EXTERNAL_AGENT_NAME, f"Decision is {intent}, proceeding... \n{task_summary}")
    writer(custom_message)

    goto = _route_from_pending_step(pending_step)

    return Command(
        goto=goto,
        update={
            "resume": False,
            "hitl_decision": hitl_decision,
            "hitl_artifact": hitl_artifact,
            "hitl_task": None,
            "messages": messages + [AIMessage(f"HITL decision: {intent} (step={pending_step}, task={task_summary})")],
        },
    )


def initialise_query(state: "InterviewPlanState") -> Command:
    runtime, writer, messages = _get_ctx(state)
    form_config = runtime.context["forms"]
    # print("form config", form_config)

    # If we arrived here after routing, consume decision if relevant
    # print(f"state {state}")
    decision = state.get("hitl_decision")
    hitl_artifact = state.get("hitl_artifact") or {}
    feedback = decision.task_summary if decision and decision.intent == "feedback" else ""

    # If user accepted the form (artifact present), persist into state and proceed
    if decision and decision.intent == "accept" and hitl_artifact:
        claim_id = hitl_artifact.get("claim_id")
        brand = hitl_artifact.get("brand")
        # interviewee = hitl_artifact.get("interviewee")
        lob = hitl_artifact.get("lob")
        investigation_type = hitl_artifact.get("investigation_type")
        investigation_scope = hitl_artifact.get("investigation_scope")
        initial_review = hitl_artifact.get("initial_review")
        additional_info = hitl_artifact.get("additional_info")
        # full_investigation = hitl_artifact.get("full_investigation")
        # additional_enquiries = hitl_artifact.get("additional_enquiries")
        print(f"investigation type {investigation_type}")

        if claim_id and investigation_type and investigation_scope and initial_review:
            return Command(
                goto="retrieve_knowledge",
                update={
                    "claim_id": claim_id,
                    "brand": brand,
                    # "interviewee": interviewee,
                    "lob": lob,
                    "investigation_type": investigation_type,
                    "investigation_scope": investigation_scope,
                    "initial_review": initial_review,
                    "additional_info": additional_info,
                    # "full_investigation": full_investigation,
                    # "additional_enquiries": additional_enquiries,
                    "hitl_decision": decision,
                    "hitl_artifact": None,
                    "pending_step": None,
                    "messages": messages + [AIMessage("Inputs captured. Proceeding to retrieve knowledge.")],
                },
            )

    # If already have required fields, proceed
    if state.get("claim_id") and state.get("lob") and state.get("investigation_type") and state.get("initial_review"):
        return Command(
            goto="retrieve_knowledge",
            update={"messages": messages +
                    [AIMessage("Inputs present. Retrieving knowledge.")], "pending_step": None},
        )

    # Otherwise: prompt user with form via HITL
    text = "Hi! I am the External Agent - I can assist you in writing instructions for external agent appointment. To begin, please proceed to fill out the form..."
    artifact = build_form_info(form_config)
    hitl_task = prepare_hitl_task(
        agent_name=EXTERNAL_AGENT_NAME,
        text=text,
        context="User must provide details.",
        state={} if use_checkpointer else {
            **state, "messages": messages + [AIMessage(text)]},
        artifact=artifact,
    )

    custom_message = prepare_thinking_message(
        EXTERNAL_AGENT_NAME, f"Waiting for form submission...")
    writer(custom_message)

    return Command(
        goto="route_interrupt",
        update={
            "pending_step": "init",
            "hitl_task": hitl_task,
            "messages": messages + [AIMessage("Awaiting form submission.")],
            "hitl_decision": None,
            "hitl_artifact": None,
        },
    )


async def _run_tool_call_async(
    call: dict,
    *,
    filters: dict,
    knowledge_endpoint: str
) -> Tuple[str, dict, Any]:
    """
    Run ONE tool call concurrently (async).
    Returns: (tool_name, args, result)
    """
    tool_name = call["name"]
    args = call.get("args", {}) or {}

    if tool_name == "query_investigation_processes":
        result = await asyncio.to_thread(
            query_investigation_processes.invoke,
            {"endpoint_name": knowledge_endpoint,
                "query": args["query"], "filters": filters},
        )
        return tool_name, args, result

    if tool_name == "think_tool":
        result = await asyncio.to_thread(
            think_tool.invoke,
            args.get("reflection", ""),
        )
        return tool_name, args, result

    if tool_name == "search_complete":
        return tool_name, args, {"ok": True}

    return tool_name, args, {"error": f"Unknown tool {tool_name}"}


async def _run_retrieval_task(
    task_key: str,
    system_prompt: str,
    prompt: str,
    llm_with_tools,
    filters: dict,
    knowledge_endpoint: str,
    max_iterations: int = 20
) -> list[Knowledge]:
    """
    Run ONE retrieval task and return extracted knowledge
    """
    prompts = [SystemMessage(content=system_prompt),
               HumanMessage(content=prompt)]

    collected: list[Knowledge] = []

    for _ in range(max_iterations):
        llm_response = await asyncio.to_thread(
            llm_with_tools.invoke,
            prompts,
            temperature=0.0,
            max_tokens=8192,
        )

        tool_calls = getattr(llm_response, "tool_calls", []) or []
        if not tool_calls:
            break

        finish = any(c["name"] == "search_complete" for c in tool_calls)
        calls_to_run = [
            c for c in tool_calls if c["name"] != "search_complete"]

        if not calls_to_run:
            break

        results = await asyncio.gather(
            *[_run_tool_call_async(c, filters=filters, knowledge_endpoint=knowledge_endpoint)
              for c in calls_to_run],
            return_exceptions=True,
        )

        for call, r in zip(calls_to_run, results):
            if isinstance(r, Exception):
                raise Exception(
                    f"Retrieval task '{task_key}' failed",
                    SIErrorCode.AGENT_ENDPOINT_FAILURE,
                ) from r

            tool_name, _, result = r
            prompts.append(
                HumanMessage(f"[Tool {tool_name}] result:\n{result}")
            )

            if tool_name == "query_investigation_processes":
                rows = (result or {}).get("knowledge", []) or []
                for row in rows:
                    collected.append(
                        Knowledge(
                            query=row.get("query", ""),
                            answer=row.get("answer", ""),
                        )
                    )

        if finish:
            break

    return collected


async def retrieve_knowledge_async(state: "InterviewPlanState") -> Command:
    runtime, writer, messages = _get_ctx(state)
    knowledge_endpoint = runtime.context["resources_endpoint_name"]

    writer(prepare_thinking_message(EXTERNAL_AGENT_NAME,
           "Retrieving knowledge on external agent..."))

    decision = state.get("hitl_decision")
    # print(f"decision {decision}")
    task_summary = decision.task_summary if decision else ""

    lob = state.get("lob", "")
    investigation_types = state.get("investigation_type", []) or []
    print(f"investigation type {investigation_types}")
    if not investigation_types or not lob:
        raise ValueError(
            f"lob or investigation_type has not been passed in state: {state}")
    initial_review = state.get("initial_review", "")

    parser = PydanticOutputParser(pydantic_object=KnowledgeSet)
    llm_with_tools = llm.bind_tools(
        [query_investigation_processes, search_complete, think_tool])

    knowledge_items: list[Knowledge] = []

    for inv_type in investigation_types:
        filters = _parse_investigation_type_to_filters(lob, inv_type)

        system_prompt = KNOWLEDGE_RETRIEVAL_SYSTEM_PROMPT

        task_results = await asyncio.gather(
            *[
                _run_retrieval_task(
                    task_key=task_key,
                    system_prompt=system_prompt,
                    prompt=KNOWLEDGE_RETRIEVAL_TASK_PROMPT.format(
                        task=task_def["task"],
                        stopping_criteria=task_def["stopping_criteria"],
                        investigation_type=inv_type,
                        # interview_case=initial_review,
                        format=parser.get_format_instructions(),
                    ),
                    llm_with_tools=llm_with_tools,
                    filters=filters,
                    knowledge_endpoint=knowledge_endpoint
                )
                for task_key, task_def in RETRIEVAL_TASKS.items()
            ]
        )

        for result in task_results:
            knowledge_items.extend(result)

    knowledge = KnowledgeSet(knowledge=knowledge_items)

    # ---- Report synthesis
    writer(prepare_thinking_message(EXTERNAL_AGENT_NAME,
           "Synthesising knowledge on external agent instruction planning..."))

    report_parser = PydanticOutputParser(pydantic_object=KnowledgeReport)
    report_system_prompt = KNOWLEDGE_REPORT_SYSTEM_PROMPT
    report_prompt = KNOWLEDGE_REPORT_PROMPT.format(
        investigation_type=", ".join(investigation_types),
        knowledge_set=knowledge.model_dump_json(),
        format=report_parser.get_format_instructions()
    )

    report_prompts = [SystemMessage(
        content=report_system_prompt), HumanMessage(content=report_prompt)]

    report_response = llm.invoke(
        input=report_prompts,
        temperature=0.0,
        max_tokens=8192,
        response_format={"type": "json_object"},
    )
    report_content = report_response.content if isinstance(
        report_response.content, str) else report_response.content[0]["text"]
    knowledge_report = report_parser.parse(report_content)

    return Command(
        goto="generate_plan",
        update={
            "knowledge": knowledge_report,
            "messages": messages + [AIMessage("Investigation processes retrieved.")],
            "pending_step": None,
        },
    )


def retrieve_knowledge(state: "InterviewPlanState") -> Command:
    """
    Sync wrapper so LangGraph node can stay def-based.
    Uses asyncio to run the parallel tool executor.
    """
    try:
        return asyncio.run(retrieve_knowledge_async(state))
    except RuntimeError:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(retrieve_knowledge_async(state))


def generate_plan(state: "InterviewPlanState") -> Command:
    runtime, writer, messages = _get_ctx(state)

    decision = state.get("hitl_decision")
    hitl_artifact = state.get("hitl_artifact") or {}
    hitl_feedback = decision.task_summary if decision and decision.intent == "feedback" else ""

    print(
        f"inside generate_plan decision {decision} \n hitl artifact {hitl_artifact}")

    # Accept => persist edited plan and proceed to finalise
    if decision and decision.intent == "accept" and hitl_artifact:
        interview_plan = hitl_artifact

        return Command(
            goto="finalise_plan",
            update={
                "interview_plans": interview_plan,
                "messages": messages + [AIMessage("Interview plan accepted.")],
                "pending_step": None,
                # "hitl_decision": None,
                "hitl_artifact": None,
            },
        )

    # Otherwise: generate/re-generate draft (feedback triggers regeneration)
    initial_review = state.get("initial_review", "")
    additional_info = state.get("additional_info", "")
    # additional_enquiries = state.get("additional_enquiries")
    # interviewee = state.get("interviewee", "")

    # interview_strategy = state.get("interview_strategy")
    # strategy_json = interview_strategy.model_dump_json(indent=2, exclude_none=True)

    knowledge_report: KnowledgeReport = state.get("knowledge")
    doc_knowledge_json = knowledge_report.model_dump_json(
        indent=2, exclude_none=True, include={"document_set"})
    print(f"Doc Knowledge json {doc_knowledge_json}")
    enquiry_knowledge_json = knowledge_report.model_dump_json(
        indent=2, exclude_none=True, include={"enquiries_rationale"})
    print(f"Enquiry Knowledge json {enquiry_knowledge_json}")
    # interview_plan_knowledge_json = knowledge_report.model_dump_json(indent=2, exclude_none=True, include={"interview_plan"})
    # print(f"Interview Plan Knowledge json {interview_plan_knowledge_json}")

    # ---- Extract previous plan version
    prev_plan: InterviewPlan = state.get("interview_plans") or {}
    prev_version = getattr(prev_plan, "version", None)
    if prev_version is None and isinstance(prev_plan, dict):
        prev_version = prev_plan.get("version", 0)
    version = prev_version + 1

    # ---- Extract feedback
    online_feedback = state.get("online_eval", {}).get("per_metric", [])
    online_feedback_block = (
        "\n\n".join(
            f"Metric: {f['metric_id']} \nSuggestions: {f['suggestions']}" for f in online_feedback)
        if online_feedback
        else ""
    )

    # feedback = (hitl_feedback or online_feedback_block).strip()
    feedback = (hitl_feedback).strip()

    max_version = 5
    if (not hitl_feedback) and online_feedback_block and prev_version >= max_version:
        # stop looping
        feedback = ""
        custom_message = prepare_thinking_message(
            EXTERNAL_AGENT_NAME, "Maximum redrafts reached...")
        writer(custom_message)

    custom_message = prepare_thinking_message(
        EXTERNAL_AGENT_NAME, f"Drafting agents for external investigation version={version}...")
    writer(custom_message)

    system_prompt = EXTERNAL_AGENT_SYSTEM_PROMPT

    # ---- Feedback path
    if feedback:
        mlflow.update_current_trace(tags={"llm_step": "plan_feedback"})
        parser = PydanticOutputParser(pydantic_object=ExternalAgentPlan)
        prompt = INTERVIEW_PLAN_FEEDBACK_PROMPT.format(
            prev_version=prev_plan.model_dump_json(
                indent=2, exclude_none=True),
            feedback=feedback,
            initial_review=initial_review,
            additional_info=additional_info,
            # knowledge=str(knowledge_json),
            # interview_strategy=str(strategy_json),
            format=parser.get_format_instructions(),
        )

        prompts = [SystemMessage(content=system_prompt),
                   HumanMessage(content=prompt)]

        response = llm.invoke(
            input=prompts,
            temperature=0.0,
            max_tokens=15104,
            response_format={"type": "json_object"},
        )
        content = response.content if isinstance(
            response.content, str) else response.content[0]["text"]
        parsed_plan: ExternalAgentPlan = parser.parse(content)

        draft_plan = ExternalAgentPlan(
            concern_set=parsed_plan,
            document_set=parsed_plan,
            enquiry_rationale_set=parsed_plan,
            version=version,
            created_at=datetime.utcnow().isoformat(),
            update_notes=None
        )

    # ---- Draft path
    else:
        mlflow.update_current_trace(tags={"llm_step": "plan_draft"})
        # key concern section
        question_parser = PydanticOutputParser(pydantic_object=KeyConcernSet)
        question_prompt = KEY_CONCERNS_DRAFT_PROMPT.format(
            # interviewee=interviewee,
            initial_review=initial_review,
            additional_info=additional_info,
            # knowledge=str(knowledge_json),
            # interview_strategy=str(strategy_json),
            format=question_parser.get_format_instructions(),
        )

        prompts = [SystemMessage(content=system_prompt),
                   HumanMessage(content=question_prompt)]

        # Response handling for concerns
        response = llm.invoke(input=prompts, temperature=0.0,
                              max_tokens=15104, response_format={"type": "json_object"})
        question_content = response.content if isinstance(
            response.content, str) else response.content[0]["text"]
        parsed_questions: KeyConcernSet = question_parser.parse(
            question_content)

        # document request section
        doc_parser = PydanticOutputParser(pydantic_object=DocRequestSet)
        doc_request_prompt = DOC_REQUEST_DRAFT_PROMPT.format(
            initial_review=initial_review,
            knowledge=str(doc_knowledge_json),
            format=doc_parser.get_format_instructions()
        )

        prompts = [SystemMessage(content=system_prompt),
                   HumanMessage(content=doc_request_prompt)]
        response = llm.invoke(input=prompts, temperature=0.0,
                              max_tokens=4000, response_format={"type": "json_object"})
        doc_content = response.content if isinstance(
            response.content, str) else response.content[0]["text"]
        parsed_doc: DocRequestSet = doc_parser.parse(doc_content)

        # additional enquiries rationale section
        enquiry_parser = PydanticOutputParser(
            pydantic_object=AdditionalEnquiriesSet)
        additional_enquiry_prompt = ADDITIONAL_ENQUIRIES_DRAFT_PROMPT.format(
            # additional_enquiries=additional_enquiries,
            initial_review=initial_review,
            knowledge=str(enquiry_knowledge_json),
            format=enquiry_parser.get_format_instructions()
        )

        prompts = [SystemMessage(content=system_prompt), HumanMessage(
            content=additional_enquiry_prompt)]
        response = llm.invoke(input=prompts, temperature=0.0,
                              max_tokens=4000, response_format={"type": "json_object"})
        enquiry_content = response.content if isinstance(
            response.content, str) else response.content[0]["text"]
        parsed_enquiries: AdditionalEnquiriesSet = enquiry_parser.parse(
            enquiry_content)

        # interview plan section (commented out in source)
        # ...

        draft_plan = ExternalAgentPlan(
            concern_set=parsed_questions,
            document_set=parsed_doc,
            enquiry_set=parsed_enquiries,
            # interview_plan=parsed_interview_plan,
            version=version,
            created_at=datetime.utcnow().isoformat(),
            update_notes=None
        )

    return Command(
        goto="online_evaluation",
        update={
            "interview_plans": draft_plan,
            # "pending_step": "plan_review",
            # "hitl_task": hitl_task,
            "messages": messages + [AIMessage("Proceeding to online evaluation.")],
            # "hitl_decision": None,
            "hitl_artifact": None,
        },
    )


def finalise_plan(state: "InterviewPlanState") -> Command:
    writer: StreamWriter = get_stream_writer()
    messages = state.get("messages", [])

    custom_message = prepare_thinking_message(
        EXTERNAL_AGENT_NAME, "Finalising instructions for external agent...")
    writer(custom_message)

    # final_strategy = state.get("interview_strategy")
    final_plan = state.get("interview_plans")
    claim_id = state.get("claim_id")
    online_eval = state.get("online_eval")
    per_metric = (online_eval.get("per_metric") or [])
    composite = (online_eval.get("composite") or {})

    artifact = build_form_final(
        claim_id=claim_id,
        # interview_strategy=final_strategy,
        interview_plan=final_plan,
        per_metric=per_metric,
        composite=composite,
        max_score=4
    )

    return Command(
        update={
            "messages": messages + [AIMessage("Final interview plan submitted. Note: You are not able to provide any further feedback in the current session.")],
            "artifact": artifact,
            "pending_step": None,
        }
    )

    graph = (
        StateGraph(InterviewPlanState)
        .add_node("initialise_query", initialise_query)
        .add_node("retrieve_knowledge", retrieve_knowledge)
        # .add_node("generate_strategy", generate_strategy)
        .add_node("generate_plan", generate_plan)
        .add_node("online_evaluation", online_evaluation)
        .add_node("finalise_plan", finalise_plan)
        .add_node("route_interrupt", route_interrupt)
        .add_edge(START, "initialise_query")
        # .add_edge("retrieve_knowledge", "generate_strategy")
        .add_edge("retrieve_knowledge", "generate_plan")
        .add_edge("finalise_plan", END)
    )

    return graph
