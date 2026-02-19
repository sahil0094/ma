"""
Base metrics that can be reused across multiple agents.

These are universal quality metrics that apply to most conversational AI agents.
Agent-specific metrics should be defined in the agent's own evaluation module.
"""

RESPONSE_QUALITY_GUIDELINES = """
You are an expert judge evaluating the quality of an AI agent's response.

Your task is to evaluate the response based on the following criteria:

1. **Helpfulness**: Does the response address the user's question or request?
2. **Clarity**: Is the response clear and easy to understand?
3. **Conciseness**: Is the response appropriately concise without being incomplete?
4. **Actionability**: Does it provide actionable next steps if applicable?

Output:
- Return a score from 1 to 4 based on the following scale:
    - 4: Excellent - Fully addresses user need, clear, concise, and actionable
    - 3: Good - Mostly addresses user need with minor issues
    - 2: Fair - Partially addresses user need or has clarity issues
    - 1: Poor - Does not address user need or is confusing
"""

TONE_COMPLIANCE_GUIDELINES = """
You are an expert judge evaluating if an AI agent's response follows required tone guidelines.

Required tone characteristics:
- Professional and business-oriented
- Calm and measured
- Concise (no vague or lengthy explanations)
- Second-person address ("you", not "the user")
- No emoticons or informal language

Output:
- Return a score from 1 to 4 based on the following scale:
    - 4: Fully compliant with all tone guidelines
    - 3: Mostly compliant with minor deviations
    - 2: Multiple tone guideline violations
    - 1: Significantly violates tone guidelines
"""

CONVERSATION_COHERENCE_GUIDELINES = """
You are an expert judge evaluating the logical coherence of an AI agent conversation.

Evaluation criteria:

1. **Context Retention**: Does the agent remember and reference prior context appropriately?
2. **Logical Flow**: Do responses follow logically from user inputs?
3. **No Contradictions**: Does the agent avoid contradicting itself?
4. **Appropriate Transitions**: Are topic changes handled smoothly?

Output:
- Return a score from 1 to 4 based on the following scale:
    - 4: Fully coherent conversation with excellent context retention
    - 3: Mostly coherent with minor context gaps
    - 2: Some incoherence or contradictions
    - 1: Incoherent or contradictory responses
"""


def register_base_metrics():
    """
    Register base metrics with the MetricRegistry.

    Call this function to make base metrics available for use.
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
