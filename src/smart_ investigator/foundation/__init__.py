"""
Offline Evaluation Framework

This package provides a generic evaluation framework for offline evaluation
of agent traces stored in MLflow.

Usage:
    from smart_investigator.foundation.evals.offline import (
        TraceRetriever,
        MetricRegistry,
        run_evaluation,
    )

    # Register metrics
    from smart_investigator.foundation.evals.offline.registry.base_metrics import register_base_metrics
    register_base_metrics()

    # Run evaluation
    results = run_evaluation(
        agent_name="my_agent",
        profiles=my_profiles,
        experiment_id="123",
        recent_hours=24,
    )
"""

from smart_investigator.foundation.evals.offline.core import (
    TraceRetriever,
    create_conversation_judge,
    create_input_output_judge,
    create_tool_call_judge,
    run_profile,
    run_evaluation,
)
from smart_investigator.foundation.evals.offline.registry import MetricRegistry

__all__ = [
    "TraceRetriever",
    "MetricRegistry",
    "create_conversation_judge",
    "create_input_output_judge",
    "create_tool_call_judge",
    "run_profile",
    "run_evaluation",
]
