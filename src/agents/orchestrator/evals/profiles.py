"""
Master Agent Evaluation Profiles

Each profile defines:
- span_name: The LangGraph node/span to evaluate
- trace_filter: MLflow filter string for trace selection
- metrics: List of metric IDs to apply

Profiles can use trace tags set by the Master Agent:
- tags.tool_called: Name of tool that was invoked
- tags.is_hitl: "true" or "false"
- tags.workflow_name: Active workflow name
- tags.workflow_finished: "true" or "false"
"""

MASTER_AGENT_PROFILES = {
    # Profile 1: General conversation quality
    # Evaluates overall response quality across all successful interactions
    "general_quality": {
        "span_name": "intent_classifier",
        "trace_filter": "status = 'SUCCESS'",
        "metrics": ["response_quality", "tone_compliance", "conversation_coherence"]
    },

    # Profile 2: Routing analysis (non-HITL tool calls)
    # Evaluates tool selection decisions and task completion
    "routing_analysis": {
        "span_name": "tool_node",
        "trace_filter": "status = 'SUCCESS' AND tags.is_hitl = 'false'",
        "metrics": ["routing_plausibility", "task_completion"]
    },

    # Profile 3: Error case analysis
    # Evaluates conversations that resulted in errors
    "error_analysis": {
        "span_name": "intent_classifier",
        "trace_filter": "status = 'ERROR'",
        "metrics": ["conversation_coherence"]
    },

    # Profile 4: HITL interaction analysis
    # Evaluates human-in-the-loop interactions specifically
    "hitl_analysis": {
        "span_name": "tool_node",
        "trace_filter": "tags.is_hitl = 'true'",
        "metrics": ["response_quality", "task_completion"]
    },

    # Profile 5: Workflow completion analysis
    # Evaluates conversations where a workflow completed
    "workflow_completion": {
        "span_name": "intent_classifier",
        "trace_filter": "tags.workflow_finished = 'true'",
        "metrics": ["task_completion", "conversation_coherence"]
    },

    # Profile 6: Workflow in-progress analysis
    # Evaluates ongoing workflow interactions
    "workflow_in_progress": {
        "span_name": "intent_classifier",
        "trace_filter": "tags.workflow_name != '' AND tags.workflow_finished = 'false'",
        "metrics": ["response_quality", "conversation_coherence"]
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
