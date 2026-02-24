import mlflow
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional

from smart_investigator.foundation.evals.offline.core.trace_retriever import TraceRetriever
from smart_investigator.foundation.evals.offline.registry.metric_registry import MetricRegistry


def _requires_tool_context(metric_ids: List[str]) -> bool:
    """
    Check if any metric in the list uses the tool_call judge type.

    Args:
        metric_ids: List of metric IDs to check

    Returns:
        True if any metric uses tool_call judge type
    """
    for metric_id in metric_ids:
        try:
            _, _, judge_type = MetricRegistry.get(metric_id)
            if judge_type == "tool_call":
                return True
        except KeyError:
            continue
    return False


def _embed_tool_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Embed tool context into the inputs column for tool_call judges.

    MLflow's make_judge only supports {{ inputs }}, {{ outputs }}, {{ expectations }},
    and {{ trace }} template variables. To provide structured tool data to judges,
    we embed the extracted tool information into the inputs field.

    Args:
        df: DataFrame with tool_spans and tool_names columns from get_traces_with_tool_context

    Returns:
        DataFrame with inputs column containing embedded tool context
    """
    df = df.copy()

    def build_enriched_input(row):
        original_input = row.get("inputs", "")
        tool_names = row.get("tool_names", [])
        tool_spans = row.get("tool_spans", [])

        return {
            "request": original_input,
            "tools_called": tool_names,
            "tool_details": tool_spans,
        }

    df["inputs"] = df.apply(build_enriched_input, axis=1)
    return df


def run_profile(
    profile: Dict,
    experiment_id: str,
    recent_hours: int,
    model: str = "azure:/gpt-4o",
    limit: int = 50,
) -> Optional[Any]:
    """
    Run evaluation for a single profile.

    Supports both trace-level and span-level evaluation:
    - If span_name is provided: evaluates specific spans within traces
    - If span_name is omitted: evaluates at trace level directly
    - If metrics use tool_call judge: enriches traces with tool span data

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
        MLflow evaluation result object, or None if no traces found
    """
    metrics = profile.get("metrics", [])
    requires_tools = _requires_tool_context(metrics)

    # Use tool-enriched retrieval if any metric needs tool context
    if requires_tools:
        traces_df = TraceRetriever.get_traces_with_tool_context(
            hours=recent_hours,
            experiment_ids=[experiment_id],
            filter_string=profile.get("trace_filter"),
        )
    else:
        traces_df = TraceRetriever.get_recent_traces(
            hours=recent_hours,
            experiment_ids=[experiment_id],
            filter_string=profile.get("trace_filter"),
        )

    if traces_df.empty:
        return None

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

    if eval_df.empty:
        return None

    # Embed tool context into inputs for tool_call judges
    if requires_tools and "tool_spans" in eval_df.columns:
        eval_df = _embed_tool_context(eval_df)

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

            if result is None:
                print(f"Skipping profile '{profile_name}': no traces found matching filter")
                mlflow.set_tag("status", "skipped")
                mlflow.set_tag("skip_reason", "no_traces_found")
                all_results[profile_name] = None
            else:
                all_results[profile_name] = result

    return all_results


def run_session_evaluation(
    agent_name: str,
    profiles: Dict[str, Dict],
    experiment_id: str,
    recent_hours: int,
    model: str = "azure:/gpt-4o",
    limit: int = 20,
    run_profiles: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run session-level evaluation across multiple profiles.

    Groups traces by session, aggregates into conversations,
    then evaluates each session as a single unit.

    Args:
        agent_name: Name of the agent being evaluated
        profiles: Dictionary of profile configurations, each with:
            - metrics: List of session-level metric IDs
        experiment_id: MLflow experiment ID containing traces
        recent_hours: Time window in hours for trace retrieval
        model: LLM model to use for judge evaluation
        limit: Maximum number of sessions to evaluate
        run_profiles: Optional list of profile names to run. If None, runs all.

    Returns:
        Dictionary mapping profile names to their evaluation results
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    sessions = TraceRetriever.get_recent_sessions(
        hours=recent_hours,
        experiment_ids=[experiment_id],
        min_traces=2
    )

    if not sessions:
        print("No sessions found with multiple traces")
        return {}

    if len(sessions) > limit:
        sessions = dict(list(sessions.items())[:limit])

    # Build evaluation data: one row per session
    eval_data = []
    for session_id, session_traces in sessions.items():
        conversation = TraceRetriever.build_session_conversation(session_traces)
        eval_data.append({
            "session_id": session_id,
            "inputs": {"conversation": conversation},  # MLflow expects dict
            "outputs": {"response": ""},  # MLflow expects dict
            "trace_count": len(session_traces),
        })

    eval_df = pd.DataFrame(eval_data)

    # Filter profiles if specific ones requested
    if run_profiles:
        profiles = {k: v for k, v in profiles.items() if k in run_profiles}

    all_results = {}

    for profile_name, profile in profiles.items():
        scorers = MetricRegistry.build_judges(profile["metrics"], model=model)
        run_name = f"{agent_name}_session_{profile_name}_{timestamp}"

        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("agent_name", agent_name)
            mlflow.set_tag("eval_type", "session")
            mlflow.set_tag("profile_name", profile_name)
            mlflow.set_tag("session_count", len(sessions))

            result = mlflow.genai.evaluate(data=eval_df, scorers=scorers)
            all_results[profile_name] = result

    return all_results
