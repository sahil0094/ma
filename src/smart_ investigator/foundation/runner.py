import mlflow
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional

from smart_investigator.foundation.evals.offline.core.trace_retriever import TraceRetriever
from smart_investigator.foundation.evals.offline.registry.metric_registry import MetricRegistry


def run_profile(
    profile: Dict,
    experiment_id: str,
    recent_hours: int,
    model: str = "azure:/gpt-4o",
    limit: int = 50,
) -> Any:
    """
    Run evaluation for a single profile.

    Supports both trace-level and span-level evaluation:
    - If span_name is provided: evaluates specific spans within traces
    - If span_name is omitted: evaluates at trace level directly

    Args:
        profile: Profile configuration with keys:
            - span_name: (Optional) Name of the span to evaluate. If omitted, evaluates traces.
            - trace_filter: MLflow filter string for traces
            - metrics: List of metric IDs to evaluate
        experiment_id: MLflow experiment ID containing traces
        recent_hours: Time window in hours for trace retrieval
        model: LLM model to use for judge evaluation
        limit: Maximum number of traces to evaluate

    Returns:
        MLflow evaluation result object
    """
    traces_df = TraceRetriever.get_recent_traces(
        hours=recent_hours,
        experiment_ids=[experiment_id],
        filter_string=profile.get("trace_filter"),
    )

    if len(traces_df) > limit:
        traces_df = traces_df.head(limit)

    span_name = profile.get("span_name")

    if span_name:
        # Span-level evaluation: extract specific spans
        trace_ids = traces_df["trace_id"].tolist()
        eval_df = TraceRetriever.get_spans_by_name(trace_ids, span_name=span_name)
    else:
        # Trace-level evaluation: use traces directly
        eval_df = traces_df

    scorers = MetricRegistry.build_judges(profile["metrics"], model=model)

    return mlflow.genai.evaluate(data=eval_df, scorers=scorers)


def run_evaluation(
    agent_name: str,
    profiles: Dict[str, Dict],
    experiment_id: str,
    recent_hours: int,
    model: str = "azure:/gpt-4o",
    limit: int = 50,
    run_profiles: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run evaluation across multiple profiles for an agent.

    Creates flat MLflow runs (one per profile) with naming:
    {agent_name}_{profile_name}_{timestamp}

    Args:
        agent_name: Name of the agent being evaluated (e.g., "master_agent")
        profiles: Dictionary of profile configurations
        experiment_id: MLflow experiment ID containing traces
        recent_hours: Time window in hours for trace retrieval
        model: LLM model to use for judge evaluation
        limit: Maximum number of traces per profile
        run_profiles: Optional list of profile names to run. If None, runs all.

    Returns:
        Dictionary mapping profile names to their evaluation results
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Filter profiles if specific ones requested
    if run_profiles:
        profiles = {k: v for k, v in profiles.items() if k in run_profiles}

    all_results = {}

    for profile_name, profile in profiles.items():
        run_name = f"{agent_name}_{profile_name}_{timestamp}"

        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("agent_name", agent_name)
            mlflow.set_tag("profile_name", profile_name)
            mlflow.set_tag("eval_type", "offline")

            result = run_profile(
                profile=profile,
                experiment_id=experiment_id,
                recent_hours=recent_hours,
                model=model,
                limit=limit
            )
            all_results[profile_name] = result

    return all_results


def run_session_evaluation(
    agent_name: str,
    profile: Dict,
    experiment_id: str,
    recent_hours: int,
    model: str = "azure:/gpt-4o",
    limit: int = 20,
) -> Any:
    """
    Run session-level evaluation.

    Groups traces by session, aggregates into conversations,
    then evaluates each session as a single unit.

    Args:
        agent_name: Name of the agent being evaluated
        profile: Profile configuration with:
            - metrics: List of session-level metric IDs
        experiment_id: MLflow experiment ID containing traces
        recent_hours: Time window in hours for trace retrieval
        model: LLM model to use for judge evaluation
        limit: Maximum number of sessions to evaluate

    Returns:
        MLflow evaluation result object
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    sessions = TraceRetriever.get_recent_sessions(
        hours=recent_hours,
        experiment_ids=[experiment_id],
        min_traces=2
    )

    if not sessions:
        print("No sessions found with multiple traces")
        return None

    if len(sessions) > limit:
        sessions = dict(list(sessions.items())[:limit])

    # Build evaluation data: one row per session
    eval_data = []
    for session_id, session_traces in sessions.items():
        conversation = TraceRetriever.build_session_conversation(session_traces)
        eval_data.append({
            "session_id": session_id,
            "inputs": conversation,  # Full session as input
            "outputs": "",  # Not used for session eval
            "trace_count": len(session_traces),
        })

    eval_df = pd.DataFrame(eval_data)
    scorers = MetricRegistry.build_judges(profile["metrics"], model=model)

    run_name = f"{agent_name}_session_{timestamp}"

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("agent_name", agent_name)
        mlflow.set_tag("eval_type", "session")
        mlflow.set_tag("profile_name", "session_quality")
        mlflow.set_tag("session_count", len(sessions))

        return mlflow.genai.evaluate(data=eval_df, scorers=scorers)
