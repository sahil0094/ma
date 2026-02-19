import mlflow
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

    Args:
        profile: Profile configuration with keys:
            - span_name: Name of the span to evaluate
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

    trace_ids = traces_df["trace_id"].tolist()
    spans_df = TraceRetriever.get_spans_by_name(
        trace_ids, span_name=profile["span_name"]
    )

    scorers = MetricRegistry.build_judges(profile["metrics"], model=model)

    return mlflow.genai.evaluate(data=spans_df, scorers=scorers)


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
    run_name = f"{agent_name}_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Filter profiles if specific ones requested
    if run_profiles:
        profiles = {k: v for k, v in profiles.items() if k in run_profiles}

    all_results = {}

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("agent_name", agent_name)
        mlflow.set_tag("eval_type", "offline")

        for profile_name, profile in profiles.items():
            with mlflow.start_run(run_name=profile_name, nested=True):
                mlflow.set_tag("profile_name", profile_name)
                mlflow.set_tag("agent_name", agent_name)

                result = run_profile(
                    profile=profile,
                    experiment_id=experiment_id,
                    recent_hours=recent_hours,
                    model=model,
                    limit=limit
                )
                all_results[profile_name] = result

    return all_results
