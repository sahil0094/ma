"""
Trace-level metrics extraction for operational monitoring.

Extracts token usage, costs, and timing metrics from MLflow traces
for monitoring LLM usage across the Smart Investigator multi-agent system.

Usage:
    pricing = AgentPricingConfig(
        model_name="gpt-4o",
        input_cost_per_1k_usd=0.0025,    # $2.50 per 1M = $0.0025 per 1K
        output_cost_per_1k_usd=0.01,      # $10.00 per 1M = $0.01 per 1K
        usd_to_aud_multiplier=1.55,
    )
    extractor = TraceMetricsExtractor(pricing)
    metrics = extractor.extract_trace_metrics(trace)
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from zoneinfo import ZoneInfo
import json
import logging
import mlflow


# Brisbane timezone (AEST, UTC+10)
BRISBANE_TZ = ZoneInfo("Australia/Brisbane")


logger = logging.getLogger(__name__)


@dataclass
class AgentPricingConfig:
    """Pricing configuration for an agent's model.

    Each agent provides this config based on the model it uses.
    Prices are in USD per 1,000 tokens, then converted to AUD.
    """
    model_name: str
    input_cost_per_1k_usd: float
    output_cost_per_1k_usd: float
    usd_to_aud_multiplier: float
    reasoning_cost_per_1k_usd: Optional[float] = None


@dataclass
class TokenMetrics:
    """Token usage metrics extracted from trace metadata."""

    input_tokens: int
    output_tokens: int
    reasoning_tokens: Optional[int]
    total_tokens: int
    model_name: Optional[str]


@dataclass
class CostMetrics:
    """Cost metrics calculated from token usage (in AUD)."""

    input_cost_aud: Decimal
    output_cost_aud: Decimal
    reasoning_cost_aud: Optional[Decimal]
    total_cost_aud: Decimal


@dataclass
class TraceMetrics:
    """Complete metrics for a single trace."""

    # Identifiers
    trace_id: str
    experiment_id: Optional[str]

    # Timestamps
    trace_timestamp: datetime

    # Token metrics
    input_tokens: int
    output_tokens: int
    reasoning_tokens: Optional[int]
    total_tokens: int

    # Cost (AUD)
    input_cost_aud: Decimal
    output_cost_aud: Decimal
    reasoning_cost_aud: Optional[Decimal]
    total_cost_aud: Decimal

    # Timing (ms)
    total_duration_ms: int

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
            "trace_timestamp": self.trace_timestamp,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
            "input_cost_aud": float(self.input_cost_aud),
            "output_cost_aud": float(self.output_cost_aud),
            "reasoning_cost_aud": float(self.reasoning_cost_aud) if self.reasoning_cost_aud else None,
            "total_cost_aud": float(self.total_cost_aud),
            "total_duration_ms": self.total_duration_ms,
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
    Calculate cost in AUD from token counts using agent pricing config.

    Pricing is per 1,000 tokens in USD, then converted to AUD.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        reasoning_tokens: Number of reasoning tokens (for reasoning models)
        pricing_config: Agent's pricing configuration

    Returns:
        CostMetrics with calculated costs in AUD
    """
    usd_to_aud = Decimal(str(pricing_config.usd_to_aud_multiplier))

    # Calculate costs in USD (tokens / 1K * cost_per_1K)
    input_cost_usd = Decimal(str(input_tokens)) / Decimal("1000") * \
        Decimal(str(pricing_config.input_cost_per_1k_usd))
    output_cost_usd = Decimal(str(output_tokens)) / Decimal("1000") * \
        Decimal(str(pricing_config.output_cost_per_1k_usd))

    # Convert to AUD
    input_cost_aud = input_cost_usd * usd_to_aud
    output_cost_aud = output_cost_usd * usd_to_aud

    # Handle reasoning cost (optional)
    reasoning_cost_aud: Optional[Decimal] = None
    if reasoning_tokens and pricing_config.reasoning_cost_per_1k_usd:
        reasoning_cost_usd = Decimal(str(reasoning_tokens)) / Decimal("1000") * \
            Decimal(str(pricing_config.reasoning_cost_per_1k_usd))
        reasoning_cost_aud = reasoning_cost_usd * usd_to_aud

    total_cost_aud = input_cost_aud + output_cost_aud + (reasoning_cost_aud or Decimal("0"))

    return CostMetrics(
        input_cost_aud=input_cost_aud,
        output_cost_aud=output_cost_aud,
        reasoning_cost_aud=reasoning_cost_aud,
        total_cost_aud=total_cost_aud,
    )


class TraceMetricsExtractor:
    """Extracts operational metrics from MLflow traces.

    Uses trace-level aggregated token usage from trace_metadata (MLflow 3.8+)
    for simpler and more robust extraction.

    Each agent instantiates this with its own pricing configuration.
    Costs are calculated in AUD.

    Example:
        pricing = AgentPricingConfig(
            model_name="gpt-4o",
            input_cost_per_1k_usd=0.0025,
            output_cost_per_1k_usd=0.01,
            usd_to_aud_multiplier=1.55,
        )
        extractor = TraceMetricsExtractor(pricing)
        metrics = extractor.extract_trace_metrics(trace)
    """

    def __init__(self, pricing_config: AgentPricingConfig):
        """
        Initialize extractor with agent's pricing configuration.

        Args:
            pricing_config: Pricing configuration from agent's config
        """
        self.pricing_config = pricing_config

    def extract_token_metrics(self, trace_info: Any) -> TokenMetrics:
        """
        Extract token metrics from trace_info.trace_metadata.

        MLflow 3.8+ provides aggregated token usage at trace level via
        the 'mlflow.trace.tokenUsage' key in trace_metadata.

        Args:
            trace_info: TraceInfo object from trace.info

        Returns:
            TokenMetrics with token counts
        """
        metadata = getattr(trace_info, "trace_metadata", {}) or {}

        # Parse mlflow.trace.tokenUsage JSON string
        token_usage_str = metadata.get("mlflow.trace.tokenUsage")
        if token_usage_str:
            try:
                token_usage = json.loads(token_usage_str)
                input_tokens = token_usage.get("input_tokens", 0)
                output_tokens = token_usage.get("output_tokens", 0)
                reasoning_tokens = token_usage.get("reasoning_tokens")
                total_tokens = token_usage.get("total_tokens", 0)

                # Get model name from trace metadata
                model_name = metadata.get("mlflow.modelId")

                return TokenMetrics(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    reasoning_tokens=reasoning_tokens,
                    total_tokens=total_tokens,
                    model_name=model_name,
                )
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse token usage JSON: {e}")

        # No LLM calls in this trace or parsing failed
        return TokenMetrics(0, 0, None, 0, None)

    def extract_trace_metrics(self, trace: Any) -> Optional[TraceMetrics]:
        """
        Extract complete metrics from a single trace.

        Uses trace-level data from TraceInfo for token usage, timing,
        and metadata (MLflow 3.8+).

        Args:
            trace: MLflow trace object

        Returns:
            TraceMetrics or None if trace is invalid
        """
        if not trace:
            return None

        info = trace.info

        # Extract token metrics from trace_info (trace-level aggregation)
        token_metrics = self.extract_token_metrics(info)

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

        # Timestamp from trace info (Brisbane timezone)
        # request_time is in milliseconds
        request_time = getattr(info, "request_time", None)
        if request_time:
            # Convert from UTC timestamp to Brisbane time
            utc_dt = datetime.fromtimestamp(request_time / 1000, tz=timezone.utc)
            trace_timestamp = utc_dt.astimezone(BRISBANE_TZ)
        else:
            trace_timestamp = datetime.now(BRISBANE_TZ)

        # Execution duration from trace-level info
        total_duration_ms = getattr(info, "execution_duration", 0) or 0

        # Status - handle TraceState enum
        state = getattr(info, "state", None)
        status = state.value if hasattr(state, "value") else str(state or "UNKNOWN")

        # Error message (from trace tags or first error span)
        error_message = self._extract_error_message(trace)

        return TraceMetrics(
            trace_id=trace_id,
            experiment_id=experiment_id,
            trace_timestamp=trace_timestamp,
            input_tokens=token_metrics.input_tokens,
            output_tokens=token_metrics.output_tokens,
            reasoning_tokens=token_metrics.reasoning_tokens,
            total_tokens=token_metrics.total_tokens,
            input_cost_aud=cost_metrics.input_cost_aud,
            output_cost_aud=cost_metrics.output_cost_aud,
            reasoning_cost_aud=cost_metrics.reasoning_cost_aud,
            total_cost_aud=cost_metrics.total_cost_aud,
            total_duration_ms=total_duration_ms,
            status=status,
            error_message=error_message,
            model_name=token_metrics.model_name,
        )

    def _extract_error_message(self, trace: Any) -> Optional[str]:
        """Extract error message from trace if status is ERROR."""
        info = trace.info
        state = getattr(info, "state", None)
        status = state.value if hasattr(state, "value") else str(state or "")

        if status != "ERROR":
            return None

        # Try to get error from trace tags
        tags = getattr(info, "tags", {}) or {}
        if "error" in tags:
            return tags["error"]

        # Try to get from span events (if spans exist)
        if hasattr(trace, "data") and trace.data and hasattr(trace.data, "spans"):
            for span in trace.data.spans or []:
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
