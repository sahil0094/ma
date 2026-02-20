"""
Master Agent Evaluation Profiles

Each profile defines:
- trace_filter: MLflow filter string for trace selection
- metrics: List of metric IDs to apply

Profiles can use trace tags set by the Master Agent:
- tags.tool_called: Name of tool that was invoked
- tags.is_hitl: "true" or "false"
- tags.workflow_name: Active workflow name
- tags.workflow_finished: "true" or "false"
"""

MASTER_AGENT_PROFILES = {
    # General quality evaluation on successful traces
    "general_quality": {
        "trace_filter": "status = 'OK'",
        "metrics": ["tone_compliance", "conversation_coherence"]
    },

    # Task completion evaluation
    "task_completion": {
        "trace_filter": "status = 'OK'",
        "metrics": ["task_completion"]
    },

    # Error trace analysis
    "error_analysis": {
        "trace_filter": "status = 'ERROR'",
        "metrics": ["conversation_coherence"]
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
