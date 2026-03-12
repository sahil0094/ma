# Databricks notebook source
# MAGIC %md
# MAGIC # Master Agent Operational Metrics Job
# MAGIC
# MAGIC **Purpose:** Extract token usage, costs, and timing metrics from MLflow traces and write to Delta table.
# MAGIC
# MAGIC **Schedule:** Daily at 2 AM UTC (0 2 * * *)
# MAGIC
# MAGIC **Owner:** Smart Investigator Team
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Parameters
# MAGIC | Parameter | Description | Required |
# MAGIC |-----------|-------------|----------|
# MAGIC | `experiment_id` | MLflow experiment ID for Master Agent | Yes |
# MAGIC | `catalog` | Unity Catalog name | Yes |
# MAGIC | `schema` | Schema name | Yes |
# MAGIC | `table_name` | Target table name | No (default: `fact_master_agent_metrics`) |
# MAGIC | `lookback_hours` | Hours to look back for traces | No (default: 26) |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# Widget parameters for Databricks Workflows
dbutils.widgets.text("experiment_id", "", "MLflow Experiment ID")
dbutils.widgets.text("catalog", "", "Unity Catalog")
dbutils.widgets.text("schema", "", "Schema")
dbutils.widgets.text("table_name", "fact_master_agent_metrics", "Table Name")
dbutils.widgets.text("lookback_hours", "26", "Lookback Hours")

# COMMAND ----------

# Get parameters
experiment_id = dbutils.widgets.get("experiment_id")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
table_name = dbutils.widgets.get("table_name")
lookback_hours = int(dbutils.widgets.get("lookback_hours"))

# Validate required parameters
if not experiment_id:
    raise ValueError("experiment_id is required")
if not catalog:
    raise ValueError("catalog is required")
if not schema:
    raise ValueError("schema is required")

print(f"Configuration:")
print(f"  Experiment ID: {experiment_id}")
print(f"  Target Table:  {catalog}.{schema}.{table_name}")
print(f"  Lookback:      {lookback_hours} hours")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Metrics Extraction Job

# COMMAND ----------

from agents.orchestrator.monitoring import run_metrics_job, MASTER_AGENT_PRICING
from agents.orchestrator.monitoring.config import MonitoringConfig

# Create configuration
config = MonitoringConfig(
    experiment_id=experiment_id,
    catalog=catalog,
    schema=schema,
    table_name=table_name,
    lookback_hours=lookback_hours,
)

print(f"Starting job with config:")
print(f"  Table path: {config.table_path}")
print(f"  Pricing: {MASTER_AGENT_PRICING.model_name} @ ${MASTER_AGENT_PRICING.input_price_per_1m}/1M input, ${MASTER_AGENT_PRICING.output_price_per_1m}/1M output")

# COMMAND ----------

# Run the job
result = run_metrics_job(config, spark=spark)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Job Results

# COMMAND ----------

# Display results
print("=" * 60)
print("JOB RESULT")
print("=" * 60)
print(f"Status:           {result.status}")
print(f"Job Run ID:       {result.job_run_id}")
print(f"Duration:         {result.duration_seconds:.2f} seconds")
print("-" * 60)
print(f"Traces Found:     {result.traces_found}")
print(f"Traces Processed: {result.traces_processed}")
print(f"Traces Failed:    {result.traces_failed}")
print(f"Rows Written:     {result.rows_written}")
print("-" * 60)
print(f"Target Table:     {result.table_path}")
print("=" * 60)

if result.error_message:
    print(f"\nERROR: {result.error_message}")

if result.failed_trace_ids:
    print(f"\nFailed Trace IDs ({len(result.failed_trace_ids)}):")
    for trace_id in result.failed_trace_ids[:10]:  # Show first 10
        print(f"  - {trace_id}")
    if len(result.failed_trace_ids) > 10:
        print(f"  ... and {len(result.failed_trace_ids) - 10} more")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validate Output

# COMMAND ----------

# Show sample of written data
if result.rows_written > 0:
    print(f"Sample data from {config.table_path}:")
    display(
        spark.table(config.table_path)
        .filter(f"job_run_id = '{result.job_run_id}'")
        .orderBy("trace_timestamp", ascending=False)
        .limit(10)
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary Statistics

# COMMAND ----------

if result.rows_written > 0:
    # Aggregate stats for this job run
    stats_df = spark.sql(f"""
        SELECT
            COUNT(*) as total_traces,
            SUM(total_tokens) as total_tokens,
            ROUND(SUM(total_cost_usd), 4) as total_cost_usd,
            ROUND(AVG(total_duration_ms), 2) as avg_duration_ms,
            ROUND(AVG(llm_duration_ms), 2) as avg_llm_duration_ms,
            COUNT(CASE WHEN status = 'OK' THEN 1 END) as success_count,
            COUNT(CASE WHEN status = 'ERROR' THEN 1 END) as error_count
        FROM {config.table_path}
        WHERE job_run_id = '{result.job_run_id}'
    """)

    print("Summary Statistics for This Job Run:")
    display(stats_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exit with Status

# COMMAND ----------

# Exit with appropriate status for workflow orchestration
if result.status == "failure":
    raise Exception(f"Job failed: {result.error_message}")
elif result.status == "partial_success":
    print(f"WARNING: Job completed with {result.traces_failed} failed traces")
    # Don't fail the job for partial success, but log warning
else:
    print("Job completed successfully")

# Return result as notebook output (for workflow chaining)
dbutils.notebook.exit(result.status)
