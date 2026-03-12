"""
Trace-level metrics extraction for operational monitoring.

Extracts token usage, costs, and timing metrics from MLflow traces
for monitoring LLM usage across the Smart Investigator multi-agent system.

Usage:
    pricing = AgentPricingConfig(
        model_name="gpt-4o",
        input_price_per_1m=2.50,
        output_price_per_1m=10.00,
    )
    extractor = TraceMetricsExtractor(pricing)
    metrics = extractor.extract_trace_metrics(trace)
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
import logging
import mlflow


logger = logging.getLogger(__name__)


@dataclass
class AgentPricingConfig:
    """Pricing configuration for an agent's model.

    Each agent provides this config based on the model it uses.
    """
    model_name: str
    input_price_per_1m: float
    output_price_per_1m: float
    reasoning_price_per_1m: Optional[float] = None


@dataclass
class TokenMetrics:
    """Token usage metrics extracted from a trace."""

    input_tokens: int
    output_tokens: int
    reasoning_tokens: Optional[int]
    total_tokens: int
    model_name: Optional[str]
    llm_duration_ms: int


@dataclass
class CostMetrics:
    """Cost metrics calculated from token usage."""

    input_cost_usd: Decimal
    output_cost_usd: Decimal
    reasoning_cost_usd: Optional[Decimal]
    total_cost_usd: Decimal


@dataclass
class TraceMetrics:
    """Complete metrics for a single trace."""

    # Identifiers
    trace_id: str
    experiment_id: Optional[str]
    agent_name: Optional[str]

    # Timestamps
    trace_timestamp: datetime

    # Token metrics
    input_tokens: int
    output_tokens: int
    reasoning_tokens: Optional[int]
    total_tokens: int

    # Cost (USD)
    input_cost_usd: Decimal
    output_cost_usd: Decimal
    reasoning_cost_usd: Optional[Decimal]
    total_cost_usd: Decimal

    # Timing (ms)
    total_duration_ms: int
    llm_duration_ms: int

    # Status
    status: str
    error_message: Optional[str]

    # Model
    model_name: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        return {
            "trace_id": self.trace_id,
            "experiment_id": self.experiment_id,
            "agent_name": self.agent_name,
            "trace_timestamp": self.trace_timestamp,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
            "input_cost_usd": float(self.input_cost_usd),
            "output_cost_usd": float(self.output_cost_usd),
            "reasoning_cost_usd": float(self.reasoning_cost_usd) if self.reasoning_cost_usd else None,
            "total_cost_usd": float(self.total_cost_usd),
            "total_duration_ms": self.total_duration_ms,
            "llm_duration_ms": self.llm_duration_ms,
            "status": self.status,
            "error_message": self.error_message,
            "model_name": self.model_name,
        }


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    reasoning_tokens: Optional[int],
    pricing_config: AgentPricingConfig,
) -> CostMetrics:
    """
    Calculate cost in USD from token counts using agent pricing config.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        reasoning_tokens: Number of reasoning tokens (for reasoning models)
        pricing_config: Agent's pricing configuration

    Returns:
        CostMetrics with calculated costs
    """
    # Calculate costs (tokens / 1M * price_per_1M)
    input_cost = Decimal(str(input_tokens)) / Decimal("1000000") * \
        Decimal(str(pricing_config.input_price_per_1m))
    output_cost = Decimal(str(output_tokens)) / Decimal("1000000") * \
        Decimal(str(pricing_config.output_price_per_1m))

    reasoning_cost = None
    if reasoning_tokens and pricing_config.reasoning_price_per_1m:
        reasoning_cost = Decimal(str(reasoning_tokens)) / Decimal("1000000") * \
            Decimal(str(pricing_config.reasoning_price_per_1m))

    total_cost = input_cost + output_cost + (reasoning_cost or Decimal("0"))

    return CostMetrics(
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
        reasoning_cost_usd=reasoning_cost,
        total_cost_usd=total_cost,
    )


class TraceMetricsExtractor:
    """Extracts operational metrics from MLflow traces.

    Each agent instantiates this with its own pricing configuration.

    Example:
        pricing = AgentPricingConfig(
            model_name="gpt-4o",
            input_price_per_1m=2.50,
            output_price_per_1m=10.00,
        )
        extractor = TraceMetricsExtractor(pricing)
        metrics = extractor.extract_trace_metrics(trace)
    """

    # Span name for Azure OpenAI LLM calls
    LLM_SPAN_NAME = "AzureChatOpenAI"
    LLM_SPAN_TYPE = "LLM"

    def __init__(self, pricing_config: AgentPricingConfig):
        """
        Initialize extractor with agent's pricing configuration.

        Args:
            pricing_config: Pricing configuration from agent's config
        """
        self.pricing_config = pricing_config

    def extract_token_metrics(self, spans: List[Any]) -> TokenMetrics:
        """
        Extract token metrics from trace spans.

        Extracts from AzureChatOpenAI spans by aggregating token usage
        from span attributes.

        Args:
            spans: List of spans from a trace

        Returns:
            TokenMetrics with token counts
        """
        return self._extract_from_llm_spans(spans)

    def _extract_from_llm_spans(self, spans: List[Any]) -> TokenMetrics:
        """Extract and aggregate tokens from AzureChatOpenAI spans."""
        total_input = 0
        total_output = 0
        total_reasoning: Optional[int] = None
        total_llm_duration_ns = 0
        model_name = None

        for span in spans:
            if not self._is_llm_span(span):
                continue

            attributes = getattr(span, "attributes", {}) or {}

            # Standard MLflow LangChain autolog format
            input_tokens = attributes.get("llm.token_usage.input_tokens")
            output_tokens = attributes.get("llm.token_usage.output_tokens")
            reasoning_tokens = attributes.get("llm.token_usage.reasoning_tokens")

            # Alternative format (prompt_tokens/completion_tokens)
            if input_tokens is None:
                input_tokens = attributes.get("llm.token_usage.prompt_tokens")
            if output_tokens is None:
                output_tokens = attributes.get("llm.token_usage.completion_tokens")

            if input_tokens is None and output_tokens is None:
                continue

            total_input += input_tokens or 0
            total_output += output_tokens or 0

            if reasoning_tokens is not None:
                total_reasoning = (total_reasoning or 0) + reasoning_tokens

            if model_name is None:
                model_name = (
                    attributes.get("llm.model_name") or
                    attributes.get("model_name") or
                    attributes.get("llm.model")
                )

            start_ns = getattr(span, "start_time_ns", None)
            end_ns = getattr(span, "end_time_ns", None)
            if start_ns and end_ns:
                total_llm_duration_ns += end_ns - start_ns

        return TokenMetrics(
            input_tokens=total_input,
            output_tokens=total_output,
            reasoning_tokens=total_reasoning,
            total_tokens=total_input + total_output + (total_reasoning or 0),
            model_name=model_name,
            llm_duration_ms=int(total_llm_duration_ns / 1_000_000),
        )

    def _is_llm_span(self, span: Any) -> bool:
        """Check if span is an AzureChatOpenAI LLM span."""
        span_name = getattr(span, "name", "")
        span_type = getattr(span, "span_type", "")

        return span_name == self.LLM_SPAN_NAME or span_type == self.LLM_SPAN_TYPE

    def extract_trace_metrics(self, trace: Any) -> Optional[TraceMetrics]:
        """
        Extract complete metrics from a single trace.

        Args:
            trace: MLflow trace object

        Returns:
            TraceMetrics or None if trace is invalid
        """
        if not trace or not trace.data or not trace.data.spans:
            return None

        info = trace.info
        spans = trace.data.spans

        # Extract token metrics from spans
        token_metrics = self.extract_token_metrics(spans)

        # Calculate costs using instance pricing config
        cost_metrics = calculate_cost(
            input_tokens=token_metrics.input_tokens,
            output_tokens=token_metrics.output_tokens,
            reasoning_tokens=token_metrics.reasoning_tokens,
            pricing_config=self.pricing_config,
        )

        # Extract trace-level info
        trace_id = info.trace_id
        experiment_id = info.experiment_id

        # Extract agent name from trace tags
        tags = getattr(info, "tags", {}) or {}
        agent_name = tags.get("workflow_name") or tags.get("agent_name")

        # Timestamp from trace info
        timestamp_ms = getattr(info, "timestamp_ms", None)
        if timestamp_ms:
            trace_timestamp = datetime.fromtimestamp(timestamp_ms / 1000)
        else:
            trace_timestamp = datetime.now()

        # Execution time
        total_duration_ms = getattr(info, "execution_time_ms", 0) or 0

        # Status
        status = str(getattr(info, "status", "UNKNOWN"))

        # Error message (from trace tags or first error span)
        error_message = self._extract_error_message(trace)

        return TraceMetrics(
            trace_id=trace_id,
            experiment_id=experiment_id,
            agent_name=agent_name,
            trace_timestamp=trace_timestamp,
            input_tokens=token_metrics.input_tokens,
            output_tokens=token_metrics.output_tokens,
            reasoning_tokens=token_metrics.reasoning_tokens,
            total_tokens=token_metrics.total_tokens,
            input_cost_usd=cost_metrics.input_cost_usd,
            output_cost_usd=cost_metrics.output_cost_usd,
            reasoning_cost_usd=cost_metrics.reasoning_cost_usd,
            total_cost_usd=cost_metrics.total_cost_usd,
            total_duration_ms=total_duration_ms,
            llm_duration_ms=token_metrics.llm_duration_ms,
            status=status,
            error_message=error_message,
            model_name=token_metrics.model_name,
        )

    def _extract_error_message(self, trace: Any) -> Optional[str]:
        """Extract error message from trace if status is ERROR."""
        info = trace.info
        status = str(getattr(info, "status", ""))

        if status != "ERROR":
            return None

        # Try to get error from trace tags
        tags = getattr(info, "tags", {}) or {}
        if "error" in tags:
            return tags["error"]

        # Try to get from span events
        for span in trace.data.spans:
            events = getattr(span, "events", []) or []
            for event in events:
                attributes = getattr(event, "attributes", {}) or {}
                if "exception.message" in attributes:
                    # Truncate long messages
                    return attributes["exception.message"][:500]

        return None

    def extract_metrics_batch(
        self,
        traces_df: "pandas.DataFrame",
    ) -> Tuple[List[TraceMetrics], List[str]]:
        """
        Extract metrics from a batch of traces with error handling.

        Args:
            traces_df: DataFrame from mlflow.search_traces()

        Returns:
            Tuple of (successful metrics, failed trace IDs)
        """
        client = mlflow.MlflowClient()
        metrics = []
        failed_trace_ids = []

        for _, trace_row in traces_df.iterrows():
            trace_id = trace_row.get("trace_id")
            if not trace_id:
                continue

            try:
                trace = client.get_trace(trace_id)
                trace_metrics = self.extract_trace_metrics(trace)

                if trace_metrics:
                    metrics.append(trace_metrics)
            except Exception as e:
                logger.warning(f"Failed to extract metrics for trace {trace_id}: {e}")
                failed_trace_ids.append(trace_id)

        return metrics, failed_trace_ids
