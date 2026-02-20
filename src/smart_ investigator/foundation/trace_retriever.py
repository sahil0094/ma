from datetime import datetime, timedelta
from typing import Optional, List, Dict
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

    # ========== Session-Level Methods ==========

    @staticmethod
    def get_session_traces(
        session_id: str,
        experiment_ids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Get all traces for a specific session, ordered by timestamp."""
        filter_string = f"attributes.`mlflow.trace.session` = '{session_id}'"
        return mlflow.search_traces(
            experiment_ids=experiment_ids,
            filter_string=filter_string,
            order_by=["attributes.timestamp_ms ASC"]
        )

    @staticmethod
    def get_recent_sessions(
        hours: int = 24,
        experiment_ids: Optional[List[str]] = None,
        min_traces: int = 2
    ) -> Dict[str, pd.DataFrame]:
        """
        Get traces grouped by session_id.

        Args:
            hours: Time window in hours
            experiment_ids: MLflow experiment IDs to search
            min_traces: Minimum traces per session (filters out single-trace sessions)

        Returns:
            Dictionary mapping session_id to DataFrame of traces in that session
        """
        traces = TraceRetriever.get_recent_traces(
            hours=hours,
            experiment_ids=experiment_ids
        )

        if traces.empty:
            return {}

        sessions = {}

        # Try to find the session column (may vary based on MLflow version)
        session_col = None
        for col in traces.columns:
            if "mlflow.trace.session" in col:
                session_col = col
                break

        if session_col and session_col in traces.columns:
            for session_id, group in traces.groupby(session_col):
                if session_id and len(group) >= min_traces:
                    sessions[session_id] = group.sort_values("timestamp")

        return sessions

    @staticmethod
    def build_session_conversation(session_traces: pd.DataFrame) -> str:
        """
        Build aggregated conversation from multiple traces for session-level eval.

        Args:
            session_traces: DataFrame of traces in a session, ordered by timestamp

        Returns:
            Formatted conversation string with all request-response pairs
        """
        conversation_parts = []

        for _, trace in session_traces.iterrows():
            inputs = trace.get("inputs", "") or ""
            outputs = trace.get("outputs", "") or ""
            conversation_parts.append(f"User: {inputs}\nAgent: {outputs}")

        return "\n\n".join(conversation_parts)

    # ========== Tool Span Methods ==========

    @staticmethod
    def get_tool_spans(trace_id: str) -> List[dict]:
        """
        Extract TOOL spans from a trace.

        Args:
            trace_id: MLflow trace ID

        Returns:
            List of tool span dictionaries with name, inputs, outputs, status
        """
        client = mlflow.MlflowClient()
        trace = client.get_trace(trace_id)

        tool_spans = []
        if trace and trace.data and trace.data.spans:
            for span in trace.data.spans:
                span_type = getattr(span, "span_type", None)
                if span_type == "TOOL":
                    tool_spans.append({
                        "span_id": span.span_id,
                        "name": span.name,
                        "inputs": span.inputs,
                        "outputs": span.outputs,
                        "status": span.status,
                    })
        return tool_spans

    @staticmethod
    def get_traces_with_tool_context(
        hours: int = 24,
        experiment_ids: Optional[List[str]] = None,
        filter_string: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get traces enriched with TOOL span data.

        Args:
            hours: Time window in hours
            experiment_ids: MLflow experiment IDs to search
            filter_string: Additional filter string

        Returns:
            DataFrame with additional columns: tool_spans, tool_names
        """
        traces = TraceRetriever.get_recent_traces(
            hours=hours,
            experiment_ids=experiment_ids,
            filter_string=filter_string
        )

        if traces.empty:
            return traces

        # Enrich with tool span data
        enriched = []
        for _, trace in traces.iterrows():
            tool_spans = TraceRetriever.get_tool_spans(trace["trace_id"])
            trace_dict = trace.to_dict()
            trace_dict["tool_spans"] = tool_spans
            trace_dict["tool_names"] = [s["name"] for s in tool_spans]
            enriched.append(trace_dict)

        return pd.DataFrame(enriched)
