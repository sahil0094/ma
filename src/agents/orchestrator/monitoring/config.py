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
# DEFAULT PRICING CONFIGURATION
# =============================================================================
# Update these values when Azure OpenAI pricing changes
# Prices are per 1,000 tokens (USD), then converted to AUD

DEFAULT_PRICING = AgentPricingConfig(
    model_name="gpt-4o",
    input_cost_per_1k_usd=0.0025,      # $2.50 per 1M = $0.0025 per 1K
    output_cost_per_1k_usd=0.01,        # $10.00 per 1M = $0.01 per 1K
    usd_to_aud_multiplier=1.55,
    reasoning_cost_per_1k_usd=None,     # Set if using reasoning models like o1/o3
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
        model_name: Model name for pricing lookup
        input_cost_per_1k_usd: Cost per 1,000 input tokens in USD
        output_cost_per_1k_usd: Cost per 1,000 output tokens in USD
        usd_to_aud_multiplier: Multiplier to convert USD to AUD
        reasoning_cost_per_1k_usd: Cost per 1,000 reasoning tokens in USD (optional)
        table_name: Delta table name for metrics
        lookback_hours: Hours to look back for traces (with buffer for late arrivals)
        batch_size: Number of traces to process per batch (for memory management)
        max_retries: Maximum retries for transient failures
        retry_delay_seconds: Delay between retries
    """
    experiment_id: str
    catalog: str
    schema: str
    model_name: str
    input_cost_per_1k_usd: float
    output_cost_per_1k_usd: float
    usd_to_aud_multiplier: float
    reasoning_cost_per_1k_usd: Optional[float] = None
    table_name: str = "fact_master_agent_metrics"
    lookback_hours: int = 26  # 24h + 2h buffer
    batch_size: int = 100
    max_retries: int = 3
    retry_delay_seconds: int = 5

    # Computed property for full table path
    @property
    def table_path(self) -> str:
        return f"{self.catalog}.{self.schema}.{self.table_name}"

    @property
    def pricing_config(self) -> AgentPricingConfig:
        """Build AgentPricingConfig from monitoring config fields."""
        return AgentPricingConfig(
            model_name=self.model_name,
            input_cost_per_1k_usd=self.input_cost_per_1k_usd,
            output_cost_per_1k_usd=self.output_cost_per_1k_usd,
            usd_to_aud_multiplier=self.usd_to_aud_multiplier,
            reasoning_cost_per_1k_usd=self.reasoning_cost_per_1k_usd,
        )

    @classmethod
    def from_env(cls) -> "MonitoringConfig":
        """Create config from environment variables.

        Environment variables:
            MASTER_AGENT_EXPERIMENT_ID: MLflow experiment ID
            MONITORING_CATALOG: Unity Catalog name
            MONITORING_SCHEMA: Schema name
            MODEL_NAME: Model name (default: gpt-4o)
            INPUT_COST_PER_1K_USD: Input cost per 1K tokens in USD
            OUTPUT_COST_PER_1K_USD: Output cost per 1K tokens in USD
            USD_TO_AUD_MULTIPLIER: USD to AUD conversion multiplier
            REASONING_COST_PER_1K_USD: Reasoning cost per 1K tokens (optional)
            MONITORING_TABLE_NAME: Table name (optional)
            MONITORING_LOOKBACK_HOURS: Lookback window (optional)
        """
        reasoning_cost = os.getenv("REASONING_COST_PER_1K_USD")
        return cls(
            experiment_id=os.environ["MASTER_AGENT_EXPERIMENT_ID"],
            catalog=os.environ["MONITORING_CATALOG"],
            schema=os.environ["MONITORING_SCHEMA"],
            model_name=os.getenv("MODEL_NAME", "gpt-4o"),
            input_cost_per_1k_usd=float(os.environ["INPUT_COST_PER_1K_USD"]),
            output_cost_per_1k_usd=float(os.environ["OUTPUT_COST_PER_1K_USD"]),
            usd_to_aud_multiplier=float(os.environ["USD_TO_AUD_MULTIPLIER"]),
            reasoning_cost_per_1k_usd=float(reasoning_cost) if reasoning_cost else None,
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

    -- Timestamps
    trace_timestamp TIMESTAMP NOT NULL,

    -- Token Metrics
    input_tokens INT,
    output_tokens INT,
    reasoning_tokens INT,
    total_tokens INT,

    -- Cost (AUD)
    input_cost_aud DOUBLE,
    output_cost_aud DOUBLE,
    reasoning_cost_aud DOUBLE,
    total_cost_aud DOUBLE,

    -- Timing (ms)
    total_duration_ms BIGINT,

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
