"""
Master Agent specific evaluation metrics.

These metrics are tailored for evaluating the Master Agent's performance,
including routing decisions, response quality with insurance domain context,
and task completion tracking.
"""

# Override base response_quality with master agent specific context
RESPONSE_QUALITY_GUIDELINES = """
You are an expert judge evaluating the quality of the Master Agent's response
in an insurance claims processing context.

Your task is to evaluate the response based on the following criteria:

1. **Helpfulness**: Does the response address the user's question or request?
2. **Clarity**: Is the response clear and easy to understand?
3. **Conciseness**: Is the response appropriately concise without being incomplete?
4. **Actionability**: Does it provide actionable next steps if applicable?
5. **Domain Appropriateness**: Is the response appropriate for insurance/claims context?

Output:
- Return a score from 1 to 4 based on the following scale:
    - 4: Excellent - Fully addresses user need, clear, concise, and actionable
    - 3: Good - Mostly addresses user need with minor issues
    - 2: Fair - Partially addresses user need or has clarity issues
    - 1: Poor - Does not address user need or is confusing
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
You are an expert judge evaluating the logical coherence of a Master Agent conversation.

Evaluation criteria:

1. **Context Retention**: Does the agent remember and reference prior context appropriately?
2. **Logical Flow**: Do responses follow logically from user inputs?
3. **No Contradictions**: Does the agent avoid contradicting itself?
4. **Appropriate Transitions**: Are topic changes handled smoothly?
5. **Workflow Awareness**: Does the agent correctly track workflow state?

Output:
- Return a score from 1 to 4 based on the following scale:
    - 4: Fully coherent conversation with excellent context retention
    - 3: Mostly coherent with minor context gaps
    - 2: Some incoherence or contradictions
    - 1: Incoherent or contradictory responses
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
You are an expert judge evaluating if the user's apparent goal was achieved
in this conversation.

Indicators of completion:
- User received the information they requested
- User confirmed satisfaction or moved on
- Workflow reached logical conclusion
- Task was handed off appropriately to another agent

Indicators of non-completion:
- Conversation ended abruptly
- User expressed frustration or confusion
- Agent indicated it couldn't help
- Workflow was interrupted without resolution

Output:
- Return one of the following values:
    - "Yes": User goal appears achieved
    - "Partial": Some progress but not fully resolved
    - "No": User goal not achieved
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
        metric_id="response_quality",
        guidelines=RESPONSE_QUALITY_GUIDELINES,
        value_type=int,
        judge_type="input_output"
    )

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
        judge_type="conversation"
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
        judge_type="conversation"
    )
