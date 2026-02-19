"""
Master Agent Offline Evaluation Module

This module provides offline evaluation capabilities for the Master Agent,
including LLM-based metrics for response quality, tone compliance,
conversation coherence, routing plausibility, and task completion.

Usage:
    from agents.orchestrator.evals import run_master_agent_evaluation

    results = run_master_agent_evaluation(
        experiment_id="your_experiment_id",
        recent_hours=24,
    )
"""

from agents.orchestrator.evals.run_evaluation import (
    run_master_agent_evaluation,
    register_master_agent_metrics,
)

__all__ = [
    "run_master_agent_evaluation",
    "register_master_agent_metrics",
]
