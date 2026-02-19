from datetime import datetime, timedelta
from typing import Optional, List
import pandas as pd
import mlflow


class TraceRetriever:
    """Utilities for retrieving traces for offline evaluation"""

    @staticmethod
    def get_recent_traces(
        hours: int = 24,
        experiment_ids: Optional[List[str]] = None,
        filter_string: Optional[str] = None
    ) -> pd.DataFrame:
        """Get traces from the last N hours"""
        start_time = datetime.now() - timedelta(hours=hours)
        timestamp_filter = f"timestamp > {int(start_time.timestamp() * 1000)}"

        if filter_string:
            timestamp_filter = f"{timestamp_filter} AND {filter_string}"

        return mlflow.search_traces(
            experiment_ids=experiment_ids,
            filter_string=timestamp_filter
        )

    @staticmethod
    def get_traces_by_date_range(
        start_date: datetime,
        end_date: datetime,
        experiment_ids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Get traces within a specific date range"""
        filter_string = (
            f"timestamp > {int(start_date.timestamp() * 1000)} AND "
            f"timestamp < {int(end_date.timestamp() * 1000)}"
        )

        return mlflow.search_traces(
            experiment_ids=experiment_ids,
            filter_string=filter_string
        )

    @staticmethod
    def get_traces_with_errors(
        hours: int = 24,
        experiment_ids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Get only failed traces for error analysis"""
        start_time = datetime.now() - timedelta(hours=hours)
        filter_string = (
            f"timestamp > {int(start_time.timestamp() * 1000)} AND "
            f"status = 'ERROR'"
        )

        return mlflow.search_traces(
            experiment_ids=experiment_ids,
            filter_string=filter_string
        )

    @staticmethod
    def get_traces_by_model(
        model_name: str,
        hours: int = 24
    ) -> pd.DataFrame:
        """Get traces for a specific model"""
        start_time = datetime.now() - timedelta(hours=hours)
        filter_string = (
            f"timestamp > {int(start_time.timestamp() * 1000)} AND "
            f"attributes.model_name = '{model_name}'"
        )

        return mlflow.search_traces(filter_string=filter_string)

    @staticmethod
    def get_annotated_traces(
        hours: int = 24,
        experiment_ids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Get traces that have been annotated with expectations"""
        traces = TraceRetriever.get_recent_traces(
            hours=hours,
            experiment_ids=experiment_ids
        )

        # Filter to only traces with assessments
        client = mlflow.MlflowClient()
        annotated_traces = []

        for _, trace_row in traces.iterrows():
            trace = client.get_trace(trace_row['trace_id'])
            if trace.data.assessments:
                annotated_traces.append(trace_row)

        return pd.DataFrame(annotated_traces)

    @staticmethod
    def get_spans_by_name(
        trace_ids: List[str],
        span_name: str
    ) -> pd.DataFrame:
        """Get specific spans from traces by span name"""
        client = mlflow.MlflowClient()
        spans_data = []

        for trace_id in trace_ids:
            trace = client.get_trace(trace_id)
            if trace and trace.data and trace.data.spans:
                for span in trace.data.spans:
                    if span.name == span_name:
                        spans_data.append({
                            "trace_id": trace_id,
                            "span_id": span.span_id,
                            "inputs": span.inputs,
                            "outputs": span.outputs,
                            "status": span.status,
                            "attributes": span.attributes,
                        })

        return pd.DataFrame(spans_data)
