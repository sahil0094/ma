from typing import Dict, Tuple, Type, List, Any, Callable
from smart_investigator.foundation.evals.offline.core.judge_factory import (
    create_conversation_judge,
    create_input_output_judge,
    create_tool_call_judge,
)


class MetricRegistry:
    """
    Dynamic metric registry for evaluation.

    Allows agents to register their own metrics with guidelines and judge types.
    Metrics are stored as (guidelines, value_type, judge_type) tuples.

    Judge types:
    - "input_output": For evaluating input/output pairs (e.g., response_quality)
    - "conversation": For evaluating full conversations (e.g., conversation_coherence)
    - "tool_call": For evaluating tool selection decisions (e.g., routing_plausibility)
    """

    _metrics: Dict[str, Tuple[str, Type, str]] = {}

    # Judge factory mapping
    _judge_factories: Dict[str, Callable] = {
        "input_output": create_input_output_judge,
        "conversation": create_conversation_judge,
        "tool_call": create_tool_call_judge,
    }

    @classmethod
    def register(
        cls,
        metric_id: str,
        guidelines: str,
        value_type: Type,
        judge_type: str = "input_output"
    ) -> None:
        """
        Register a new metric.

        Args:
            metric_id: Unique identifier for the metric
            guidelines: Evaluation guidelines/rubric for the LLM judge
            value_type: Return type (int for 1-4 scale, bool for Yes/No, str for categories)
            judge_type: One of "input_output", "conversation", or "tool_call"
        """
        if judge_type not in cls._judge_factories:
            raise ValueError(
                f"Unknown judge_type: {judge_type}. "
                f"Must be one of: {list(cls._judge_factories.keys())}"
            )
        cls._metrics[metric_id] = (guidelines, value_type, judge_type)

    @classmethod
    def get(cls, metric_id: str) -> Tuple[str, Type, str]:
        """Get metric configuration by ID."""
        if metric_id not in cls._metrics:
            raise KeyError(f"Metric '{metric_id}' not registered")
        return cls._metrics[metric_id]

    @classmethod
    def list_metrics(cls) -> List[str]:
        """List all registered metric IDs."""
        return list(cls._metrics.keys())

    @classmethod
    def build_judges(cls, metric_ids: List[str], model: str = "azure:/gpt-4o") -> List[Any]:
        """
        Build MLflow LLM judges from metric IDs.

        Args:
            metric_ids: List of metric IDs to build judges for
            model: LLM model to use for evaluation

        Returns:
            List of MLflow judge objects

        Raises:
            ValueError: If any metric ID is not registered
        """
        judges = []
        unknown = [m for m in metric_ids if m not in cls._metrics]
        if unknown:
            raise ValueError(
                f"Unknown metric ids: {unknown}. "
                f"Registered: {list(cls._metrics.keys())}"
            )

        for metric_id in metric_ids:
            guidelines, value_type, judge_type = cls._metrics[metric_id]
            judge_factory = cls._judge_factories[judge_type]
            judges.append(
                judge_factory(
                    name=metric_id,
                    guidelines=guidelines,
                    feedback_value_type=value_type,
                    model=model,
                )
            )
        return judges

    @classmethod
    def clear(cls) -> None:
        """Clear all registered metrics. Useful for testing."""
        cls._metrics.clear()
