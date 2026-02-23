# Offline Evaluation System - Technical Documentation

## Overview

The Offline Evaluation System provides a framework for evaluating agent performance using MLflow traces and LLM-as-a-Judge scoring. It enables automated quality assessment across multiple dimensions (tone, coherence, routing, task completion) without manual review.

**Key Features:**
- Profile-based evaluation with customizable metrics
- Support for trace-level and session-level evaluation
- Dynamic metric registration via registry pattern
- Graceful handling of empty datasets
- MLflow integration for tracking and visualization

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        run_evaluation.py                            │
│                    (Agent-Specific Entry Point)                     │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           profiles.py                               │
│              (Defines evaluation profiles & filters)                │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Foundation Layer                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │    runner    │  │   registry   │  │     trace_retriever      │  │
│  │     .py      │◄─┤     .py      │  │          .py             │  │
│  └──────┬───────┘  └──────┬───────┘  └────────────┬─────────────┘  │
│         │                 │                       │                 │
│         │                 ▼                       │                 │
│         │          ┌──────────────┐               │                 │
│         │          │judge_factory │               │                 │
│         │          │     .py      │               │                 │
│         │          └──────────────┘               │                 │
└─────────┼─────────────────────────────────────────┼─────────────────┘
          │                                         │
          ▼                                         ▼
┌─────────────────────┐                 ┌─────────────────────────────┐
│   mlflow.genai      │                 │    mlflow.search_traces     │
│    .evaluate()      │                 │                             │
└─────────────────────┘                 └─────────────────────────────┘
```

---

## Core Components

### 1. MetricRegistry

Central registry for evaluation metrics using class-level storage.

**Location:** `smart_investigator/foundation/evals/offline/registry/metric_registry.py`

| Method | Description |
|--------|-------------|
| `register(metric_id, guidelines, value_type, judge_type)` | Register a new metric |
| `get(metric_id)` | Retrieve metric configuration |
| `list_metrics()` | List all registered metric IDs |
| `build_judges(metric_ids, model)` | Build MLflow judges from metric IDs |
| `clear_metrics()` | Clear all registered metrics (for testing) |

**Judge Types:**

| Type | Use Case | Template Variables |
|------|----------|-------------------|
| `input_output` | Request/response evaluation | `{{inputs}}`, `{{outputs}}` |
| `conversation` | Full conversation context | `{{conversation}}` |
| `tool_call` | Tool/routing decisions | `{{trace}}` |

---

### 2. Profiles

Profiles define what traces to evaluate and which metrics to apply.

**Location:** `agents/orchestrator/evals/profiles.py`

**Profile Structure:**

```python
{
    "profile_name": {
        "trace_filter": "MLflow filter string",  # e.g., "status = 'OK'"
        "metrics": ["metric_id_1", "metric_id_2"],
        "aggregation": "session"  # Optional: for session-level eval
    }
}
```

**Available Profiles:**

| Profile | Filter | Metrics | Level |
|---------|--------|---------|-------|
| `quality` | `status = 'OK'` | tone_compliance, conversation_coherence, task_completion | Trace |
| `routing` | `status = 'OK'` | routing_plausibility, tool_output_utilization | Trace |
| `session_quality` | (all) | session_goal_achievement, cross_turn_coherence | Session |
| `error_analysis` | `status = 'ERROR'` | conversation_coherence | Trace |

---

### 3. TraceRetriever

Utility class for fetching traces from MLflow.

**Location:** `smart_investigator/foundation/evals/offline/core/trace_retriever.py`

**Key Methods:**

| Method | Description |
|--------|-------------|
| `get_recent_traces(hours, experiment_ids, filter_string)` | Get traces from last N hours |
| `get_traces_by_date_range(start_date, end_date, experiment_ids)` | Get traces within date range |
| `get_traces_with_errors(hours, experiment_ids)` | Get failed traces only |
| `get_spans_by_name(trace_ids, span_name)` | Extract specific spans from traces |
| `get_recent_sessions(hours, experiment_ids, min_traces)` | Get traces grouped by session |
| `build_session_conversation(session_traces)` | Aggregate session into conversation string |
| `get_tool_spans(trace_id)` | Extract TOOL spans from a trace |

---

### 4. Judge Factory

Creates MLflow LLM judges from guidelines.

**Location:** `smart_investigator/foundation/evals/offline/core/judge_factory.py`

| Factory Function | Judge Type | Use Case |
|------------------|------------|----------|
| `create_input_output_judge()` | input_output | Tone, quality, task completion |
| `create_conversation_judge()` | conversation | Multi-turn coherence |
| `create_tool_call_judge()` | tool_call | Routing decisions |

---

## Metric Definitions

Metrics are defined with evaluation guidelines (rubrics) that the LLM judge follows.

**Location:** `agents/orchestrator/evals/metrics/master_agent_metrics.py`

### Example Metric Definition

```python
TONE_COMPLIANCE_GUIDELINES = """
You are an expert judge evaluating if the Master Agent's response follows
the required tone guidelines for a professional insurance platform.

Required tone characteristics:
- Professional and business-oriented
- Calm and measured (even when delivering difficult news)
- Concise (no vague or lengthy explanations)
- Second-person address ("you", not "the user")
- No emoticons or informal language

Output:
- Return a score from 1 to 4 based on the following scale:
    - 4: Fully compliant with all tone guidelines
    - 3: Mostly compliant with minor deviations
    - 2: Multiple tone guideline violations
    - 1: Significantly violates tone guidelines
"""
```

### Metric Value Types

| Value Type | Output Format | Example Metrics |
|------------|---------------|-----------------|
| `int` | 1-4 scale | tone_compliance, conversation_coherence |
| `str` | Categorical | task_completion ("Yes", "Partial", "No") |
| `bool` | True/False | (custom metrics) |

---

## Usage

### Running Evaluation via CLI

```bash
python -m agents.orchestrator.evals.run_evaluation \
    --experiment-id YOUR_EXPERIMENT_ID \
    --recent-hours 24 \
    --profiles quality routing \
    --limit 50
```

**CLI Arguments:**

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--experiment-id` | Yes | - | MLflow experiment ID |
| `--recent-hours` | No | 24 | Time window for trace retrieval |
| `--model` | No | azure:/gpt-4o | LLM model for judges |
| `--limit` | No | 50 | Max traces per profile |
| `--profiles` | No | all | Specific profiles to run |
| `--tools-config` | No | auto | Path to tools_config directory |
| `--list-profiles` | No | - | List available profiles and exit |

### Running Evaluation Programmatically

```python
from agents.orchestrator.evals.run_evaluation import run_master_agent_evaluation

results = run_master_agent_evaluation(
    experiment_id="my_experiment",
    recent_hours=24,
    profile_names=["quality", "routing"],
    limit=50
)

for profile_name, result in results.items():
    if result is None:
        print(f"{profile_name}: Skipped (no traces found)")
    else:
        print(f"{profile_name}: {result.metrics}")
```

---

## Integration Guide

### Step 1: Define Your Metrics

Create a metrics file for your agent:

```python
# agents/your_agent/evals/metrics/your_agent_metrics.py

YOUR_METRIC_GUIDELINES = """
You are an expert judge evaluating [specific aspect].

Criteria:
- Criterion 1
- Criterion 2

Output:
- Return a score from 1 to 4:
    - 4: Excellent
    - 3: Good
    - 2: Fair
    - 1: Poor
"""

def register_your_agent_metrics():
    from smart_investigator.foundation.evals.offline.registry.metric_registry import MetricRegistry

    MetricRegistry.register(
        metric_id="your_metric",
        guidelines=YOUR_METRIC_GUIDELINES,
        value_type=int,
        judge_type="input_output"
    )
```

### Step 2: Define Your Profiles

```python
# agents/your_agent/evals/profiles.py

YOUR_AGENT_PROFILES = {
    "quality": {
        "trace_filter": "status = 'OK'",
        "metrics": ["your_metric", "tone_compliance"]
    },
    "errors": {
        "trace_filter": "status = 'ERROR'",
        "metrics": ["conversation_coherence"]
    }
}

def get_profiles(profile_names=None):
    if profile_names is None:
        return YOUR_AGENT_PROFILES
    return {k: v for k, v in YOUR_AGENT_PROFILES.items() if k in profile_names}
```

### Step 3: Create Entry Point

```python
# agents/your_agent/evals/run_evaluation.py

from smart_investigator.foundation.evals.offline import run_evaluation, MetricRegistry
from agents.your_agent.evals.metrics.your_agent_metrics import register_your_agent_metrics
from agents.your_agent.evals.profiles import YOUR_AGENT_PROFILES, get_profiles

def run_your_agent_evaluation(
    experiment_id: str,
    recent_hours: int = 24,
    profile_names: list = None,
):
    # Clear previous metrics
    MetricRegistry.clear_metrics()

    # Register your metrics
    register_your_agent_metrics()

    # Get profiles
    profiles = get_profiles(profile_names)

    # Run evaluation
    return run_evaluation(
        agent_name="your_agent",
        profiles=profiles,
        experiment_id=experiment_id,
        recent_hours=recent_hours,
    )
```

### Step 4: Ensure Traces Are Tagged

For session-level evaluation, ensure your agent sets the session attribute:

```python
import mlflow

# When creating a trace
mlflow.update_current_trace(
    attributes={"mlflow.trace.session": session_id}
)
```

---

## Adding New Metrics

### Trace-Level Metric

```python
MetricRegistry.register(
    metric_id="response_accuracy",
    guidelines="""
    Evaluate if the response contains accurate information.

    Score 1-4:
    - 4: Completely accurate
    - 3: Minor inaccuracies
    - 2: Several inaccuracies
    - 1: Mostly inaccurate
    """,
    value_type=int,
    judge_type="input_output"
)
```

### Session-Level Metric

```python
MetricRegistry.register(
    metric_id="goal_achievement",
    guidelines="""
    Evaluate if the user achieved their goal across the session.

    Output: "Yes", "Partial", or "No"
    """,
    value_type=str,
    judge_type="conversation"
)
```

### Tool/Routing Metric

```python
MetricRegistry.register(
    metric_id="routing_accuracy",
    guidelines="""
    Evaluate if the correct tool was selected for the user's request.

    Available tools: {tool_descriptions}

    Output: "Yes", "No", or "Unclear"
    """,
    value_type=str,
    judge_type="tool_call"
)
```

---

## Empty Dataset Handling

The system gracefully handles profiles that match no traces:

- **Trace-level:** If `trace_filter` matches no traces, the profile is skipped
- **Session-level:** If no sessions with `min_traces` exist, the profile is skipped
- **MLflow tags:** Skipped profiles are tagged with `status=skipped` and `skip_reason=no_traces_found`
- **Return value:** `None` is returned for skipped profiles

```python
results = run_evaluation(...)

for profile, result in results.items():
    if result is None:
        print(f"{profile}: No traces found")
    else:
        print(f"{profile}: {result.metrics}")
```

---

## MLflow Integration

### Viewing Results

1. Open MLflow UI
2. Navigate to your experiment
3. Look for runs named: `{agent_name}_{profile_name}_{timestamp}`
4. Each run contains:
   - Evaluation metrics (aggregated scores)
   - Per-trace scores in artifacts
   - Tags: `agent_name`, `profile_name`, `eval_type`

### Filter String Syntax

Use MLflow filter syntax for `trace_filter`:

| Filter | Description |
|--------|-------------|
| `status = 'OK'` | Successful traces only |
| `status = 'ERROR'` | Failed traces only |
| `tags.workflow_name = 'claims'` | Specific workflow |
| `attributes.model_name = 'gpt-4'` | Specific model |

---

## Best Practices

1. **Keep guidelines concise:** LLM judges work best with clear, specific rubrics
2. **Use appropriate value types:** `int` for scales, `str` for categorical, `bool` for binary
3. **Choose correct judge type:**
   - `input_output` for single-turn evaluation
   - `conversation` for multi-turn context
   - `tool_call` for routing decisions
4. **Set reasonable limits:** Start with `limit=50` to avoid long evaluation times
5. **Register metrics fresh:** Always call `clear_metrics()` before registering to avoid stale state
6. **Tag traces properly:** Ensure production traces have proper tags for filtering

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "Empty dataset" error | No traces match filter | System now skips gracefully; check your filter |
| MetricRegistry returns function | Import path mismatch | Import directly from `metric_registry.py`, not `__init__.py` |
| Metrics not found | Metrics not registered | Call `register_*_metrics()` before `run_evaluation()` |
| Session eval returns empty | No multi-turn sessions | Ensure `mlflow.trace.session` attribute is set |

---

## File Structure

```
agents/orchestrator/evals/
├── __init__.py
├── run_evaluation.py          # Entry point
├── profiles.py                # Profile definitions
└── metrics/
    ├── __init__.py
    └── master_agent_metrics.py  # Metric definitions

smart_investigator/foundation/evals/offline/
├── core/
│   ├── runner.py              # Evaluation orchestration
│   ├── trace_retriever.py     # MLflow trace fetching
│   └── judge_factory.py       # LLM judge creation
└── registry/
    └── metric_registry.py     # Central metric registry
```
