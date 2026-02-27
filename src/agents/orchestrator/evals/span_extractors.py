"""
Master Agent span extractors for span-level evaluation.

These extractors are specific to the Master Agent's trace structure
and extract relevant spans for routing and tone evaluation.
"""

import re
from typing import Optional, List
import pandas as pd
import mlflow

from smart_investigator.foundation.evals.offline.core.trace_retriever import TraceRetriever


def get_routing_spans(
    hours: int = 24,
    experiment_ids: Optional[List[str]] = None,
    filter_string: Optional[str] = None
) -> pd.DataFrame:
    """
    Extract Responses spans from intent_classifier for routing evaluation.

    Only extracts from intent_classifier spans that have AzureChatOpenAI as child
    (indicating an actual LLM routing decision was made).

    Args:
        hours: Time window in hours
        experiment_ids: MLflow experiment IDs to search
        filter_string: Additional filter string

    Returns:
        DataFrame with columns: trace_id, span_id, inputs, outputs
    """
    traces = TraceRetriever.get_recent_traces(
        hours=hours,
        experiment_ids=experiment_ids,
        filter_string=filter_string
    )

    if traces.empty:
        return pd.DataFrame()

    client = mlflow.MlflowClient()
    routing_data = []

    for _, trace_row in traces.iterrows():
        trace = client.get_trace(trace_row["trace_id"])
        if not trace or not trace.data or not trace.data.spans:
            continue

        spans = trace.data.spans
        span_map = {s.span_id: s for s in spans}

        for span in spans:
            # Find Responses spans
            if span.name != "Responses":
                continue

            # Check parent chain: Responses -> AzureChatOpenAI -> intent_classifier
            parent = span_map.get(span.parent_id)
            if not parent or parent.name != "AzureChatOpenAI":
                continue

            grandparent = span_map.get(parent.parent_id)
            if not grandparent or grandparent.name != "intent_classifier":
                continue

            routing_data.append({
                "trace_id": trace_row["trace_id"],
                "span_id": span.span_id,
                "inputs": span.inputs,
                "outputs": span.outputs,
            })

    return pd.DataFrame(routing_data)


def get_tone_spans(
    hours: int = 24,
    experiment_ids: Optional[List[str]] = None,
    filter_string: Optional[str] = None
) -> pd.DataFrame:
    """
    Extract MasterAgent responses from human_in_the_loop spans for tone evaluation.

    Extracts text from Interrupt exceptions where sender == 'MasterAgent'.
    Uses span.events[0].attributes to access exception.message and sender.

    Args:
        hours: Time window in hours
        experiment_ids: MLflow experiment IDs to search
        filter_string: Additional filter string

    Returns:
        DataFrame with columns: trace_id, span_id, inputs (user request), outputs (agent text)
    """
    traces = TraceRetriever.get_recent_traces(
        hours=hours,
        experiment_ids=experiment_ids,
        filter_string=filter_string
    )

    if traces.empty:
        return pd.DataFrame()

    client = mlflow.MlflowClient()
    tone_data = []

    for _, trace_row in traces.iterrows():
        trace = client.get_trace(trace_row["trace_id"])
        if not trace or not trace.data or not trace.data.spans:
            continue

        for span in trace.data.spans:
            if span.name != "human_in_the_loop":
                continue

            # Check if span has events
            events = getattr(span, "events", None)
            if not events or len(events) == 0:
                continue

            # Get attributes from first event
            attributes = getattr(events[0], "attributes", {})
            if not attributes:
                continue

            # Check sender - only evaluate MasterAgent responses
            sender = attributes.get("sender", "")
            if sender != "MasterAgent":
                continue

            # Get exception message containing the Interrupt
            exception_message = attributes.get("exception.message", "")
            if not exception_message or "Interrupt" not in exception_message:
                continue

            # Extract text from interrupt string
            agent_text = _extract_interrupt_text(exception_message)

            if agent_text:
                tone_data.append({
                    "trace_id": trace_row["trace_id"],
                    "span_id": span.span_id,
                    "inputs": span.inputs,
                    "outputs": agent_text,
                })

    return pd.DataFrame(tone_data)


def _extract_interrupt_text(message: str) -> Optional[str]:
    """
    Extract text from Interrupt exception message string.

    Args:
        message: The exception message string containing Interrupt

    Returns:
        Extracted text or None if not found
    """
    # Pattern 1: text="..." (double quotes - most common)
    match = re.search(r'text="((?:[^"\\]|\\.)*)"', message)
    if match:
        text = match.group(1)
        return text.replace('\\"', '"').replace("\\n", "\n")

    # Pattern 2: text='...' (single quotes)
    match = re.search(r"text='((?:[^'\\]|\\.)*)'", message)
    if match:
        text = match.group(1)
        return text.replace("\\'", "'").replace("\\n", "\n")

    return None
