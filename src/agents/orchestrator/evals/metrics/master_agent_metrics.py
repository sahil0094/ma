"""
Master Agent specific evaluation metrics.

These metrics are tailored for evaluating the Master Agent's performance,
including tone compliance, conversation coherence, and task completion tracking.
"""

TONE_COMPLIANCE_GUIDELINES = """
You are an expert judge evaluating if the Master Agent's response follows
the required tone guidelines for a professional insurance platform.

Required tone characteristics:
- Professional and business-oriented
- Calm and measured (even when delivering difficult news)
- Concise (no vague or lengthy explanations)
- Second-person address ("you", not "the user")
- No emoticons or informal language
- Use "draft" instead of "denial" when applicable

Output:
- Return a score from 1 to 4 based on the following scale:
    - 4: Fully compliant with all tone guidelines
    - 3: Mostly compliant with minor deviations
    - 2: Multiple tone guideline violations
    - 1: Significantly violates tone guidelines
"""

CONVERSATION_COHERENCE_GUIDELINES = """
You are an expert judge evaluating the logical coherence of a Master Agent's response
to a user request.

Evaluation criteria:

1. **Relevance**: Does the response directly address the user's request?
2. **Logical Flow**: Does the response follow logically from the input?
3. **Completeness**: Does the response fully answer what was asked?
4. **Clarity**: Is the response clear and easy to understand?
5. **No Confusion**: Does the response avoid introducing confusion or ambiguity?

Output:
- Return a score from 1 to 4 based on the following scale:
    - 4: Fully coherent response that directly addresses the request
    - 3: Mostly coherent with minor gaps or tangents
    - 2: Some incoherence or partially addresses the request
    - 1: Incoherent or fails to address the request
"""

ROUTING_PLAUSIBILITY_GUIDELINES = """
You are an expert judge evaluating if the Master Agent's tool/agent routing
decision was plausible given the user's request.

Evaluation criteria:

1. **Intent Match**: Does the user's apparent intent align with the selected tool's purpose?
2. **Obvious Mismatch**: Is there a clear disconnect (e.g., user asks about claims, routed to billing)?
3. **Argument Quality**: Are the tool arguments reasonable for the request?

Available Tools and their purposes:
{tool_descriptions}

Output:
- Return one of the following values:
    - "Yes": Routing is plausible given user intent
    - "No": Clear mismatch between user intent and selected tool
    - "Unclear": Cannot determine if routing is appropriate

Note: You cannot definitively say routing was "correct" without ground truth.
Only evaluate if the decision was reasonable and plausible.
"""

TASK_COMPLETION_GUIDELINES = """
You are an expert judge evaluating if the user's request was fulfilled
by the agent's response.

Indicators of completion:
- Response provides the information or action the user requested
- Response confirms task was completed or handed off appropriately
- Response gives a clear resolution or next steps

Indicators of non-completion:
- Response does not address the user's request
- Response indicates inability to help without alternative
- Response is incomplete or cuts off
- Response creates more confusion than clarity

Output:
- Return one of the following values:
    - "Yes": User request appears fulfilled by the response
    - "Partial": Some progress but not fully resolved
    - "No": User request not fulfilled
"""


def register_master_agent_metrics(tool_descriptions: str = ""):
    """
    Register Master Agent metrics with the MetricRegistry.

    Args:
        tool_descriptions: Formatted string of available tools and descriptions.
                          Will be injected into ROUTING_PLAUSIBILITY_GUIDELINES.

    Call this function before running evaluation to make metrics available.
    """
    from smart_investigator.foundation.evals.offline.registry.metric_registry import MetricRegistry

    MetricRegistry.register(
        metric_id="tone_compliance",
        guidelines=TONE_COMPLIANCE_GUIDELINES,
        value_type=int,
        judge_type="input_output"
    )

    MetricRegistry.register(
        metric_id="conversation_coherence",
        guidelines=CONVERSATION_COHERENCE_GUIDELINES,
        value_type=int,
        judge_type="input_output"
    )

    # Inject tool descriptions into routing plausibility guidelines
    routing_guidelines = ROUTING_PLAUSIBILITY_GUIDELINES.format(
        tool_descriptions=tool_descriptions if tool_descriptions else "No tool descriptions provided."
    )
    MetricRegistry.register(
        metric_id="routing_plausibility",
        guidelines=routing_guidelines,
        value_type=str,  # Returns "Yes", "No", or "Unclear"
        judge_type="tool_call"
    )

    MetricRegistry.register(
        metric_id="task_completion",
        guidelines=TASK_COMPLETION_GUIDELINES,
        value_type=str,  # Returns "Yes", "Partial", or "No"
        judge_type="input_output"
    )
