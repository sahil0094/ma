"""
Master Agent Operational Metrics Batch Job.

Extracts token usage, costs, and timing metrics from MLflow traces
and writes them to a Delta table for operational monitoring.

Usage (Databricks notebook):
    from agents.orchestrator.monitoring import run_metrics_job
    from agents.orchestrator.monitoring.config import MonitoringConfig

    config = MonitoringConfig(
        experiment_id="<your_experiment_id>",
        catalog="your_catalog",
        schema="your_schema",
    )
    result = run_metrics_job(config)
    print(result)

Schedule: Daily at 2 AM UTC (0 2 * * *)
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from zoneinfo import ZoneInfo
import logging
import time
import uuid


# Brisbane timezone (AEST, UTC+10)
BRISBANE_TZ = ZoneInfo("Australia/Brisbane")

import mlflow
import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    LongType,
    TimestampType,
    DoubleType,
    DateType,
)
from pyspark.sql.functions import lit, col
from delta.tables import DeltaTable

from smart_investigator.foundation.monitoring.trace_metrics_extractor import (
    TraceMetricsExtractor,
    TraceMetrics,
)
from agents.orchestrator.monitoring.config import MonitoringConfig


logger = logging.getLogger(__name__)


# =============================================================================
# DELTA TABLE SCHEMA
# =============================================================================

METRICS_SCHEMA = StructType([
    # Identifiers
    StructField("trace_id", StringType(), False),
    StructField("experiment_id", StringType(), True),
    # Timestamps
    StructField("trace_timestamp", TimestampType(), False),
    # Token Metrics
    StructField("input_tokens", IntegerType(), True),
    StructField("output_tokens", IntegerType(), True),
    StructField("reasoning_tokens", IntegerType(), True),
    StructField("total_tokens", IntegerType(), True),
    # Cost (AUD)
    StructField("input_cost_aud", DoubleType(), True),
    StructField("output_cost_aud", DoubleType(), True),
    StructField("reasoning_cost_aud", DoubleType(), True),
    StructField("total_cost_aud", DoubleType(), True),
    # Timing (ms)
    StructField("total_duration_ms", LongType(), True),
    # Status
    StructField("status", StringType(), True),
    StructField("error_message", StringType(), True),
    # Model
    StructField("model_name", StringType(), True),
    # Job Metadata
    StructField("job_run_id", StringType(), True),
    StructField("extracted_at", TimestampType(), True),
    # Partitioning
    StructField("trace_date", DateType(), False),
])


# =============================================================================
# JOB RESULT
# =============================================================================

@dataclass
class JobResult:
    """Result of a metrics extraction job run."""
    status: str  # "success", "partial_success", "failure"
    job_run_id: str
    experiment_id: str
    table_path: str
    traces_found: int
    traces_processed: int
    traces_failed: int
    rows_written: int
    job_start: datetime
    job_end: datetime
    duration_seconds: float
    error_message: Optional[str] = None
    failed_trace_ids: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "job_run_id": self.job_run_id,
            "experiment_id": self.experiment_id,
            "table_path": self.table_path,
            "traces_found": self.traces_found,
            "traces_processed": self.traces_processed,
            "traces_failed": self.traces_failed,
            "rows_written": self.rows_written,
            "job_start": self.job_start.isoformat(),
            "job_end": self.job_end.isoformat(),
            "duration_seconds": self.duration_seconds,
            "error_message": self.error_message,
            "failed_trace_ids": self.failed_trace_ids,
        }


# =============================================================================
# MASTER AGENT METRICS JOB
# =============================================================================

class MasterAgentMetricsJob:
    """Production-ready batch job for extracting Master Agent metrics.

    Features:
    - Idempotent writes via MERGE (safe to re-run)
    - Batch processing for memory efficiency
    - Retry logic for transient failures
    - Comprehensive error handling and logging
    - Job metadata tracking
    """

    def __init__(
        self,
        config: MonitoringConfig,
        spark: Optional[SparkSession] = None,
    ):
        """
        Initialize the metrics job.

        Args:
            config: Monitoring configuration (includes pricing)
            spark: SparkSession (creates one if not provided)
        """
        self.config = config
        self.spark = spark or SparkSession.builder.getOrCreate()
        self.extractor = TraceMetricsExtractor(config.pricing_config)
        self.job_run_id = str(uuid.uuid4())

    def run(self) -> JobResult:
        """
        Execute the metrics extraction job.

        Returns:
            JobResult with job execution details
        """
        job_start = datetime.now()
        logger.info(f"Starting metrics job {self.job_run_id}")
        logger.info(f"Experiment ID: {self.config.experiment_id}")
        logger.info(f"Target table: {self.config.table_path}")
        logger.info(f"Lookback: {self.config.lookback_hours} hours")

        try:
            # Step 1: Fetch traces from MLflow
            traces_df = self._fetch_traces()
            traces_found = len(traces_df)
            logger.info(f"Found {traces_found} traces")

            if traces_found == 0:
                return self._create_result(
                    status="success",
                    job_start=job_start,
                    traces_found=0,
                    traces_processed=0,
                    traces_failed=0,
                    rows_written=0,
                )

            # Step 2: Extract metrics in batches
            all_metrics, all_failed = self._extract_metrics_in_batches(traces_df)
            traces_processed = len(all_metrics)
            traces_failed = len(all_failed)
            logger.info(f"Extracted {traces_processed} metrics, {traces_failed} failed")

            if traces_processed == 0:
                return self._create_result(
                    status="failure" if traces_failed > 0 else "success",
                    job_start=job_start,
                    traces_found=traces_found,
                    traces_processed=0,
                    traces_failed=traces_failed,
                    rows_written=0,
                    failed_trace_ids=all_failed,
                    error_message="No metrics extracted" if traces_failed > 0 else None,
                )

            # Step 3: Write to Delta table
            rows_written = self._write_to_delta(all_metrics)
            logger.info(f"Wrote {rows_written} rows to {self.config.table_path}")

            # Determine final status
            status = "success" if traces_failed == 0 else "partial_success"

            return self._create_result(
                status=status,
                job_start=job_start,
                traces_found=traces_found,
                traces_processed=traces_processed,
                traces_failed=traces_failed,
                rows_written=rows_written,
                failed_trace_ids=all_failed if all_failed else None,
            )

        except Exception as e:
            logger.exception(f"Job failed with error: {e}")
            return self._create_result(
                status="failure",
                job_start=job_start,
                traces_found=0,
                traces_processed=0,
                traces_failed=0,
                rows_written=0,
                error_message=str(e),
            )

    def _fetch_traces(self) -> pd.DataFrame:
        """Fetch traces from MLflow for the lookback window."""
        start_time = datetime.now() - timedelta(hours=self.config.lookback_hours)
        timestamp_ms = int(start_time.timestamp() * 1000)
        filter_string = f"timestamp > {timestamp_ms}"

        logger.info(f"Fetching traces since {start_time}")

        return mlflow.search_traces(
            experiment_ids=[self.config.experiment_id],
            filter_string=filter_string,
        )

    def _extract_metrics_in_batches(
        self,
        traces_df: pd.DataFrame,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Extract metrics from traces in batches for memory efficiency.

        Returns:
            Tuple of (list of metric records, list of failed trace IDs)
        """
        all_records = []
        all_failed = []
        client = mlflow.MlflowClient()
        extracted_at = datetime.now(BRISBANE_TZ)

        total_traces = len(traces_df)
        batch_size = self.config.batch_size

        for batch_start in range(0, total_traces, batch_size):
            batch_end = min(batch_start + batch_size, total_traces)
            batch_df = traces_df.iloc[batch_start:batch_end]
            logger.info(f"Processing batch {batch_start}-{batch_end} of {total_traces}")

            for _, trace_row in batch_df.iterrows():
                trace_id = trace_row.get("trace_id")
                if not trace_id:
                    continue

                record = self._extract_single_trace(
                    client, trace_id, extracted_at
                )
                if record:
                    all_records.append(record)
                else:
                    all_failed.append(trace_id)

        return all_records, all_failed

    def _extract_single_trace(
        self,
        client: mlflow.MlflowClient,
        trace_id: str,
        extracted_at: datetime,
    ) -> Optional[Dict[str, Any]]:
        """
        Extract metrics from a single trace with retry logic.

        Returns:
            Metric record dict or None if extraction failed
        """
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                trace = client.get_trace(trace_id)
                metrics = self.extractor.extract_trace_metrics(trace)

                if metrics:
                    record = metrics.to_dict()
                    # Add job metadata
                    record["job_run_id"] = self.job_run_id
                    record["extracted_at"] = extracted_at
                    # Add partition column
                    record["trace_date"] = metrics.trace_timestamp.date()
                    return record

                return None

            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    logger.warning(
                        f"Attempt {attempt + 1} failed for trace {trace_id}: {e}. Retrying..."
                    )
                    time.sleep(self.config.retry_delay_seconds)
                else:
                    logger.error(
                        f"All {self.config.max_retries} attempts failed for trace {trace_id}: {e}"
                    )

        return None

    def _write_to_delta(self, records: List[Dict[str, Any]]) -> int:
        """
        Write metrics to Delta table with idempotent MERGE.

        Returns:
            Number of rows written/updated
        """
        if not records:
            return 0

        # Convert Decimal to float for Spark compatibility
        for record in records:
            for key in ["input_cost_aud", "output_cost_aud", "reasoning_cost_aud", "total_cost_aud"]:
                if record.get(key) is not None:
                    record[key] = float(record[key])

        # Create DataFrame
        df = self.spark.createDataFrame(records, schema=METRICS_SCHEMA)

        # Check if table exists
        table_exists = self._table_exists()

        if not table_exists:
            logger.info(f"Creating table {self.config.table_path}")
            df.write.format("delta") \
                .partitionBy("trace_date") \
                .saveAsTable(self.config.table_path)
            return len(records)

        # MERGE for idempotency (upsert)
        logger.info(f"Merging {len(records)} records into {self.config.table_path}")

        delta_table = DeltaTable.forName(self.spark, self.config.table_path)

        (
            delta_table.alias("target")
            .merge(
                df.alias("source"),
                "target.trace_id = source.trace_id"
            )
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )

        return len(records)

    def _table_exists(self) -> bool:
        """Check if the target table exists."""
        try:
            return self.spark.catalog.tableExists(self.config.table_path)
        except Exception:
            return False

    def _create_result(
        self,
        status: str,
        job_start: datetime,
        traces_found: int,
        traces_processed: int,
        traces_failed: int,
        rows_written: int,
        error_message: Optional[str] = None,
        failed_trace_ids: Optional[List[str]] = None,
    ) -> JobResult:
        """Create a JobResult with computed fields."""
        job_end = datetime.now()
        duration = (job_end - job_start).total_seconds()

        result = JobResult(
            status=status,
            job_run_id=self.job_run_id,
            experiment_id=self.config.experiment_id,
            table_path=self.config.table_path,
            traces_found=traces_found,
            traces_processed=traces_processed,
            traces_failed=traces_failed,
            rows_written=rows_written,
            job_start=job_start,
            job_end=job_end,
            duration_seconds=duration,
            error_message=error_message,
            failed_trace_ids=failed_trace_ids,
        )

        logger.info(f"Job completed: {result.to_dict()}")
        return result


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def run_metrics_job(
    config: MonitoringConfig,
    spark: Optional[SparkSession] = None,
) -> JobResult:
    """
    Run the Master Agent metrics extraction job.

    This is the main entry point for running the job from a Databricks notebook
    or workflow.

    Args:
        config: Monitoring configuration with experiment ID, table details, and pricing
        spark: SparkSession (creates one if not provided)

    Returns:
        JobResult with execution details

    Example:
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
    """
    job = MasterAgentMetricsJob(config, spark)
    return job.run()
