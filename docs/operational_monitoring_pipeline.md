# Operational Monitoring Pipeline

## Overview

The operational monitoring pipeline extracts token usage, costs, and timing metrics from MLflow traces for monitoring LLM usage across the Smart Investigator multi-agent system. The architecture consists of two layers:

1. **Foundation Layer** - Reusable trace metrics extraction (`trace_metrics_extractor.py`)
2. **Agent-Specific Layer** - Job orchestration per agent (e.g., `metrics_job.py` for Master Agent)

---

## Foundation: Trace Metrics Extractor

**Location:** `smart_investigator/foundation/monitoring/trace_metrics_extractor.py`

This module provides the core extraction logic reusable by all agent operational pipelines.

### Data Models

| Class | Purpose |
|-------|---------|
| `AgentPricingConfig` | Pricing configuration per agent (model name, costs per 1K tokens in USD, USD to AUD multiplier) |
| `TokenMetrics` | Token counts extracted from trace (input, output, reasoning, total, model name) |
| `CostMetrics` | Calculated costs in AUD (input, output, reasoning, total) |
| `TraceMetrics` | Complete metrics for a single trace (identifiers, timestamps, tokens, costs, timing, status) |

### TraceMetricsExtractor Class

The main class that each agent instantiates with its pricing configuration.

**Initialization:**

```python
pricing = AgentPricingConfig(
    model_name="gpt-4o",
    input_cost_per_1k_usd=0.0025,    # $2.50 per 1M
    output_cost_per_1k_usd=0.01,      # $10.00 per 1M
    usd_to_aud_multiplier=1.55,
)
extractor = TraceMetricsExtractor(pricing)
```

### Key Methods

#### extract_trace_metrics(trace) -> TraceMetrics

Extracts complete metrics from a single MLflow trace object. This is the primary method called by agent jobs.

**Extraction sources (MLflow 3.8+ trace-level data):**

| Metric | Source in TraceInfo |
|--------|---------------------|
| `trace_id` | `info.trace_id` |
| `experiment_id` | `info.experiment_id` |
| `trace_timestamp` | `info.request_time` (ms to datetime) |
| `input_tokens` | `info.trace_metadata['mlflow.trace.tokenUsage']` (JSON) |
| `output_tokens` | `info.trace_metadata['mlflow.trace.tokenUsage']` (JSON) |
| `total_tokens` | `info.trace_metadata['mlflow.trace.tokenUsage']` (JSON) |
| `total_duration_sec` | `info.execution_duration` (ms to seconds) |
| `status` | `info.state` |
| `model_name` | `info.trace_metadata['mlflow.modelId']` |

#### calculate_cost (static method)

Static method for cost calculation. Can be used independently without instantiating the extractor.

**Signature:**

```python
calculate_cost(input_tokens, output_tokens, reasoning_tokens, pricing_config) -> CostMetrics
```

**Formula:**

```
cost_aud = (tokens / 1000) * cost_per_1k_usd * usd_to_aud_multiplier
```

**Usage:**

```python
# Via instance
cost = extractor.calculate_cost(1000, 500, None, pricing)

# Or directly (no instance needed)
cost = TraceMetricsExtractor.calculate_cost(1000, 500, None, pricing)
```

### Output: TraceMetrics Fields

| Field | Type | Description |
|-------|------|-------------|
| `trace_id` | str | Unique trace identifier |
| `experiment_id` | str | MLflow experiment ID |
| `trace_timestamp` | datetime | When the trace was created (Brisbane TZ) |
| `input_tokens` | int | Number of input tokens |
| `output_tokens` | int | Number of output tokens |
| `reasoning_tokens` | int (optional) | Reasoning tokens (for o1/o3 models) |
| `total_tokens` | int | Total token count |
| `input_cost_aud` | Decimal | Input cost in AUD |
| `output_cost_aud` | Decimal | Output cost in AUD |
| `reasoning_cost_aud` | Decimal (optional) | Reasoning cost in AUD |
| `total_cost_aud` | Decimal | Total cost in AUD |
| `total_duration_sec` | float | Total trace execution time in seconds |
| `status` | str | Trace status (OK, ERROR, etc.) |
| `error_message` | str (optional) | Error message if status is ERROR |
| `model_name` | str | Model used (e.g., gpt-4o) |

---

## Master Agent Metrics Job

**Location:** `agents/orchestrator/monitoring/metrics_job.py`

Production-ready batch job for extracting Master Agent metrics from MLflow traces and writing to Delta table.

### Features

- **Idempotent writes** - Uses MERGE (upsert) so safe to re-run
- **Batch processing** - Processes traces in configurable batches for memory efficiency
- **Retry logic** - Configurable retries with delay for transient failures
- **Job metadata tracking** - Each run gets a unique `job_run_id`
- **Comprehensive logging** - Progress and error tracking

### Job Flow

```
+-------------------------------------------------------------+
|                     run_metrics_job()                       |
+-------------------------------------------------------------+
                            |
                            v
+-------------------------------------------------------------+
|  Step 1: Fetch Traces                                       |
|  - Query MLflow for traces in lookback window               |
|  - Filter: timestamp > (now - lookback_hours)               |
+-------------------------------------------------------------+
                            |
                            v
+-------------------------------------------------------------+
|  Step 2: Extract Metrics in Batches                         |
|  - Process traces in batches (default: 100)                 |
|  - For each trace:                                          |
|    - Fetch full trace via client.get_trace()                |
|    - Extract metrics via TraceMetricsExtractor              |
|    - Add job metadata (job_run_id, extracted_at)            |
|    - Retry on failure (default: 3 attempts)                 |
+-------------------------------------------------------------+
                            |
                            v
+-------------------------------------------------------------+
|  Step 3: Write to Delta Table                               |
|  - MERGE on trace_id (insert new, update existing)          |
|  - Partitioned by trace_date                                |
+-------------------------------------------------------------+
                            |
                            v
+-------------------------------------------------------------+
|  Return JobResult                                           |
|  - status: success / partial_success / failure              |
|  - traces_found, traces_processed, traces_failed            |
|  - rows_written, duration_seconds                           |
+-------------------------------------------------------------+
```

### Usage

```python
from agents.orchestrator.monitoring import run_metrics_job
from agents.orchestrator.monitoring.config import MonitoringConfig

config = MonitoringConfig(
    experiment_id="123456789",
    catalog="prod_catalog",
    schema="smart_investigator",
    model_name="gpt-4o",
    input_cost_per_1k_usd=0.0025,
    output_cost_per_1k_usd=0.01,
    usd_to_aud_multiplier=1.55,
)

result = run_metrics_job(config)

if result.status == "success":
    print(f"Processed {result.traces_processed} traces")
else:
    print(f"Job failed: {result.error_message}")
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `experiment_id` | required | MLflow experiment ID to query |
| `catalog` | required | Unity Catalog name |
| `schema` | required | Schema name |
| `table_name` | `fact_master_agent_metrics` | Delta table name |
| `lookback_hours` | 26 | Hours to look back (24h + 2h buffer) |
| `batch_size` | 100 | Traces per batch |
| `max_retries` | 3 | Retry attempts per trace |
| `retry_delay_seconds` | 5 | Delay between retries |

### JobResult

The job returns a `JobResult` dataclass with execution details:

| Field | Description |
|-------|-------------|
| `status` | `success`, `partial_success`, or `failure` |
| `job_run_id` | Unique identifier for this job run |
| `traces_found` | Total traces found in MLflow |
| `traces_processed` | Successfully processed traces |
| `traces_failed` | Failed trace extractions |
| `rows_written` | Rows written to Delta table |
| `duration_seconds` | Total job execution time |
| `failed_trace_ids` | List of trace IDs that failed (if any) |

---

## Adding Monitoring for a New Agent

To add operational monitoring for a new agent:

1. **Create agent-specific config** with pricing for the model used
2. **Create agent-specific job** similar to `MasterAgentMetricsJob`
3. **Reuse `TraceMetricsExtractor`** from foundation layer
4. **Configure MLflow experiment ID** for the agent

The foundation layer handles all the trace parsing and cost calculation - agent jobs only need to handle orchestration (batching, retries, writing).
