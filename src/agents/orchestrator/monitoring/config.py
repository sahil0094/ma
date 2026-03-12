"""
Configuration for Master Agent operational monitoring.

This module defines pricing configuration and monitoring settings
for the Master Agent. Update these values when model pricing changes.
"""

from dataclasses import dataclass, field
from typing import Optional
import os

from smart_investigator.foundation.monitoring.trace_metrics_extractor import (
    AgentPricingConfig,
)


# =============================================================================
# PRICING CONFIGURATION
# =============================================================================
# Update these values when Azure OpenAI pricing changes
# Prices are per 1 million tokens (USD)

MASTER_AGENT_PRICING = AgentPricingConfig(
    model_name="gpt-4o",
    input_price_per_1m=2.50,
    output_price_per_1m=10.00,
    reasoning_price_per_1m=None,  # Set if using reasoning models like o1/o3
)


# =============================================================================
# MLFLOW EXPERIMENT CONFIGURATION
# =============================================================================
# The MLflow experiment ID where Master Agent traces are logged
# Override via environment variable for different environments

MASTER_AGENT_EXPERIMENT_ID = os.getenv(
    "MASTER_AGENT_EXPERIMENT_ID",
    ""  # Set default experiment ID here
)


# =============================================================================
# MONITORING JOB CONFIGURATION
# =============================================================================

@dataclass
class MonitoringConfig:
    """Configuration for the metrics extraction batch job.

    Attributes:
        experiment_id: MLflow experiment ID to query traces from
        catalog: Unity Catalog name
        schema: Schema name within catalog
        table_name: Delta table name for metrics
        lookback_hours: Hours to look back for traces (with buffer for late arrivals)
        batch_size: Number of traces to process per batch (for memory management)
        max_retries: Maximum retries for transient failures
        retry_delay_seconds: Delay between retries
    """
    experiment_id: str
    catalog: str
    schema: str
    table_name: str = "fact_master_agent_metrics"
    lookback_hours: int = 26  # 24h + 2h buffer
    batch_size: int = 100
    max_retries: int = 3
    retry_delay_seconds: int = 5

    # Computed property for full table path
    @property
    def table_path(self) -> str:
        return f"{self.catalog}.{self.schema}.{self.table_name}"

    @classmethod
    def from_env(cls) -> "MonitoringConfig":
        """Create config from environment variables.

        Environment variables:
            MASTER_AGENT_EXPERIMENT_ID: MLflow experiment ID
            MONITORING_CATALOG: Unity Catalog name
            MONITORING_SCHEMA: Schema name
            MONITORING_TABLE_NAME: Table name (optional)
            MONITORING_LOOKBACK_HOURS: Lookback window (optional)
        """
        return cls(
            experiment_id=os.environ["MASTER_AGENT_EXPERIMENT_ID"],
            catalog=os.environ["MONITORING_CATALOG"],
            schema=os.environ["MONITORING_SCHEMA"],
            table_name=os.getenv("MONITORING_TABLE_NAME", "fact_master_agent_metrics"),
            lookback_hours=int(os.getenv("MONITORING_LOOKBACK_HOURS", "26")),
        )


# =============================================================================
# DELTA TABLE SCHEMA (for reference)
# =============================================================================
# This DDL can be used to manually create the table if needed

CREATE_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS {catalog}.{schema}.{table_name} (
    -- Identifiers
    trace_id STRING NOT NULL,
    experiment_id STRING,
    agent_name STRING,

    -- Timestamps
    trace_timestamp TIMESTAMP NOT NULL,

    -- Token Metrics
    input_tokens INT,
    output_tokens INT,
    reasoning_tokens INT,
    total_tokens INT,

    -- Cost (USD)
    input_cost_usd DECIMAL(10, 6),
    output_cost_usd DECIMAL(10, 6),
    reasoning_cost_usd DECIMAL(10, 6),
    total_cost_usd DECIMAL(10, 6),

    -- Timing (ms)
    total_duration_ms BIGINT,
    llm_duration_ms BIGINT,

    -- Status
    status STRING,
    error_message STRING,

    -- Model
    model_name STRING,

    -- Job Metadata
    job_run_id STRING,
    extracted_at TIMESTAMP,

    -- Partitioning
    trace_date DATE NOT NULL
)
USING DELTA
PARTITIONED BY (trace_date)
TBLPROPERTIES (
    'delta.autoOptimize.optimizeWrite' = 'true',
    'delta.autoOptimize.autoCompact' = 'true'
);
"""
