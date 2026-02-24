"""
Master Agent Evaluation Profiles

Each profile defines:
- trace_filter: MLflow filter string for trace selection
- metrics: List of metric IDs to apply
- aggregation: (Optional) "session" for session-level evaluation

Profiles can use trace tags and metadata set by the Master Agent:

Tags:
- tags.workflow_name: Active workflow name
- tags.workflow_finished: "true" or "false"

Metadata:
- attributes.mlflow.trace.session: Session ID for grouping traces

Note: Tool calls are captured in TOOL spans by MLflow autolog, not in tags.
"""

MASTER_AGENT_PROFILES = {
    # Trace-level quality evaluation (tone, coherence, task completion)
    "quality": {
        "trace_filter": "status = 'OK'",
        "metrics": ["tone_compliance", "conversation_coherence", "task_completion"]
    },

    # Routing quality with tool analysis
    "routing": {
        "trace_filter": "status = 'OK'",
        "metrics": ["routing_plausibility", "tool_output_utilization"]
    },

    # Session-level quality (requires session aggregation)
    "session_quality": {
        "aggregation": "session",  # Indicates session-level evaluation
        "metrics": ["session_goal_achievement", "cross_turn_coherence"]
    },

    # Error trace analysis for debugging
    "error_analysis": {
        "trace_filter": "status = 'ERROR'",
        "metrics": ["error_root_cause"]
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
