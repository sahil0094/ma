"""
Master Agent Evaluation Profiles

Each profile defines:
- span_name: (Optional) The LangGraph node/span to evaluate. If omitted, evaluates at trace level.
- trace_filter: MLflow filter string for trace selection
- metrics: List of metric IDs to apply

Evaluation Levels:
- Trace-level: Omit span_name to evaluate entire traces
- Span-level: Provide span_name to evaluate specific spans (e.g., intent_classifier, tool_node)

Profiles can use trace tags set by the Master Agent:
- tags.tool_called: Name of tool that was invoked
- tags.is_hitl: "true" or "false"
- tags.workflow_name: Active workflow name
- tags.workflow_finished: "true" or "false"
"""

MASTER_AGENT_PROFILES = {
    # ============================================
    # TRACE-LEVEL PROFILES (no span_name)
    # ============================================

    # End-to-end conversation quality
    "e2e_quality": {
        "trace_filter": "status = 'SUCCESS'",
        "metrics": ["conversation_coherence", "task_completion"]
    },

    # Full conversation tone analysis
    "e2e_tone": {
        "trace_filter": "status = 'SUCCESS'",
        "metrics": ["tone_compliance"]
    },

    # Completed workflow trace analysis
    "workflow_e2e": {
        "trace_filter": "tags.workflow_finished = 'true'",
        "metrics": ["task_completion", "conversation_coherence"]
    },

    # ============================================
    # SPAN-LEVEL PROFILES (with span_name)
    # ============================================

    # Intent classifier response quality
    "ic_quality": {
        "span_name": "intent_classifier",
        "trace_filter": "status = 'SUCCESS'",
        "metrics": ["response_quality", "tone_compliance"]
    },

    # Routing analysis (non-HITL tool calls)
    "routing_analysis": {
        "span_name": "tool_node",
        "trace_filter": "status = 'SUCCESS' AND tags.is_hitl = 'false'",
        "metrics": ["routing_plausibility"]
    },

    # Error case analysis
    "error_analysis": {
        "span_name": "intent_classifier",
        "trace_filter": "status = 'ERROR'",
        "metrics": ["conversation_coherence"]
    },

    # HITL interaction analysis
    "hitl_analysis": {
        "span_name": "tool_node",
        "trace_filter": "tags.is_hitl = 'true'",
        "metrics": ["response_quality"]
    },

    # Workflow in-progress span analysis
    "workflow_in_progress": {
        "span_name": "intent_classifier",
        "trace_filter": "tags.workflow_name != '' AND tags.workflow_finished = 'false'",
        "metrics": ["response_quality"]
    },
}


def get_profiles(profile_names: list = None) -> dict:
    """
    Get evaluation profiles.

    Args:
        profile_names: Optional list of profile names to retrieve.
                      If None, returns all profiles.

    Returns:
        Dictionary of profile configurations.
    """
    if profile_names is None:
        return MASTER_AGENT_PROFILES

    return {
        name: MASTER_AGENT_PROFILES[name]
        for name in profile_names
        if name in MASTER_AGENT_PROFILES
    }
