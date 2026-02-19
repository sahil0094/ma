"""
Master Agent Offline Evaluation Entry Point

This module provides the main entry point for running offline evaluation
on Master Agent traces stored in MLflow.

Usage:
    python -m agents.orchestrator.evals.run_evaluation \\
        --experiment-id YOUR_EXPERIMENT_ID \\
        --recent-hours 24 \\
        --profiles general_quality routing_analysis
"""

import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any

from smart_investigator.foundation.evals.offline import run_evaluation, MetricRegistry
from agents.orchestrator.evals.metrics.master_agent_metrics import register_master_agent_metrics
from agents.orchestrator.evals.profiles import MASTER_AGENT_PROFILES, get_profiles


def load_tool_descriptions(tools_config_path: Optional[Path] = None) -> str:
    """
    Load tool descriptions from tools_config YAML files at runtime.

    Args:
        tools_config_path: Path to tools_config directory.
                          If None, uses default path relative to this file.

    Returns:
        Formatted string of tool names and descriptions.
    """
    if tools_config_path is None:
        # Default: look for tools_config in the orchestrator directory
        tools_config_path = Path(__file__).parent.parent / "tools_config"

    if not tools_config_path.exists():
        return "No tools configuration found."

    descriptions = []
    for yaml_file in sorted(tools_config_path.glob("*.yaml")):
        try:
            with open(yaml_file) as f:
                config = yaml.safe_load(f)
                if config:
                    name = config.get("name", yaml_file.stem)
                    desc = config.get("description", "No description available")
                    descriptions.append(f"- {name}: {desc}")
        except Exception as e:
            descriptions.append(f"- {yaml_file.stem}: (Error loading: {e})")

    return "\n".join(descriptions) if descriptions else "No tools found."


def register_master_agent_metrics_with_tools(tools_config_path: Optional[Path] = None) -> None:
    """
    Register Master Agent metrics with tool descriptions injected.

    Args:
        tools_config_path: Optional path to tools_config directory.
    """
    tool_descriptions = load_tool_descriptions(tools_config_path)
    register_master_agent_metrics(tool_descriptions=tool_descriptions)


def run_master_agent_evaluation(
    experiment_id: str,
    recent_hours: int = 24,
    model: str = "azure:/gpt-4o",
    limit: int = 50,
    profile_names: Optional[List[str]] = None,
    tools_config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run offline evaluation for the Master Agent.

    Args:
        experiment_id: MLflow experiment ID containing Master Agent traces
        recent_hours: Time window in hours for trace retrieval (default: 24)
        model: LLM model to use for evaluation (default: azure:/gpt-4o)
        limit: Maximum traces per profile (default: 50)
        profile_names: Optional list of profile names to run. If None, runs all.
        tools_config_path: Optional path to tools_config directory for routing metric.

    Returns:
        Dictionary mapping profile names to their evaluation results.

    Example:
        results = run_master_agent_evaluation(
            experiment_id="my_experiment",
            recent_hours=24,
            profile_names=["general_quality", "routing_analysis"]
        )

        for profile, result in results.items():
            print(f"{profile}: {result}")
    """
    # Clear any previously registered metrics
    MetricRegistry.clear()

    # Register metrics with tool descriptions
    register_master_agent_metrics_with_tools(tools_config_path)

    # Get profiles to run
    profiles = get_profiles(profile_names)

    if not profiles:
        raise ValueError(
            f"No valid profiles found. Available: {list(MASTER_AGENT_PROFILES.keys())}"
        )

    # Run evaluation
    return run_evaluation(
        agent_name="master_agent",
        profiles=profiles,
        experiment_id=experiment_id,
        recent_hours=recent_hours,
        model=model,
        limit=limit,
        run_profiles=profile_names,
    )


def main():
    """CLI entry point for running Master Agent evaluation."""
    parser = argparse.ArgumentParser(
        description="Run offline evaluation for the Master Agent"
    )
    parser.add_argument(
        "--experiment-id",
        required=True,
        help="MLflow experiment ID containing traces"
    )
    parser.add_argument(
        "--recent-hours",
        type=int,
        default=24,
        help="Time window in hours for trace retrieval (default: 24)"
    )
    parser.add_argument(
        "--model",
        default="azure:/gpt-4o",
        help="LLM model for evaluation (default: azure:/gpt-4o)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum traces per profile (default: 50)"
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=list(MASTER_AGENT_PROFILES.keys()),
        help="Specific profiles to run. If not specified, runs all."
    )
    parser.add_argument(
        "--tools-config",
        type=Path,
        help="Path to tools_config directory"
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List available profiles and exit"
    )

    args = parser.parse_args()

    if args.list_profiles:
        print("Available profiles:")
        for name, config in MASTER_AGENT_PROFILES.items():
            print(f"\n{name}:")
            print(f"  span_name: {config['span_name']}")
            print(f"  trace_filter: {config['trace_filter']}")
            print(f"  metrics: {', '.join(config['metrics'])}")
        return

    results = run_master_agent_evaluation(
        experiment_id=args.experiment_id,
        recent_hours=args.recent_hours,
        model=args.model,
        limit=args.limit,
        profile_names=args.profiles,
        tools_config_path=args.tools_config,
    )

    print("\n=== Evaluation Results ===")
    for profile_name, result in results.items():
        print(f"\n{profile_name}:")
        print(f"  {result}")


if __name__ == "__main__":
    main()
