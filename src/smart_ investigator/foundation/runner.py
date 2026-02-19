import yaml
import mlflow
from datetime import datetime
from typing import Dict

from smart_investigator.foundation.tools.interview_plan.evaluation.utils import TraceRetriever
from smart_investigator.foundation.tools.interview_plan.evaluation.build_judge import build_judges


def run_profile(
    profile: Dict,
    experiment_id: str,
    recent_hours: int,
    model: str = "azure:/gpt-4o",
    limit: int = 50,
):
    traces_df = TraceRetriever.get_recent_traces(
        hours=recent_hours,
        experiment_ids=[experiment_id],
        filter_string=profile["trace_filter"],
    )

    if len(traces_df) > limit:
        traces_df = traces_df.head(limit)

    trace_ids = traces_df["trace_id"].tolist()
    spans_df = TraceRetriever.get_spans_by_name(
        trace_ids, span_name=profile["span_name"])

    scorers = build_judges(profile["metrics"], model=model)

    return mlflow.genai.evaluate(data=spans_df, scorers=scorers)


def run_evaluation(
    profiles: Dict,
    experiment_id: str,
    recent_hours: int,
    model: str = "azure:/gpt-4o",
    limit: int = 50
):
    run_name = f"interview_plan_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    all_results = {}

    with mlflow.start_run(run_name=run_name):
        for profile_name, profile in profiles.items():
            with mlflow.start_run(run_name=profile_name, nested=True):
                mlflow.set_tag("profile_name", profile_name)
                result = run_profile(
                    profile=profile,
                    experiment_id=experiment_id,
                    recent_hours=recent_hours,
                    model=model,
                    limit=limit
                )
                all_results[profile_name] = result

    return all_results
