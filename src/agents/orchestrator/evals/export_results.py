"""
Export evaluation results to Unity Catalog Delta tables.
"""

from datetime import datetime
from typing import Any, Dict
import json

import pandas as pd
from pyspark.sql import SparkSession
from delta.tables import DeltaTable


def export_to_delta(
    results: Dict[str, Any],
    experiment_id: str,
    table_name: str,
    catalog: str = "main",
    schema: str = "evaluation",
) -> None:
    """
    Export all evaluation results to a Unity Catalog Delta table.

    Uses MERGE to upsert - overwrites existing entries for same trace_id + metric_name.

    Args:
        results: Dictionary mapping profile_name -> EvaluationResult
        experiment_id: MLflow experiment ID
        table_name: Delta table name
        catalog: Unity Catalog name
        schema: Schema/database name
    """
    eval_timestamp = datetime.now()
    records = []

    non_metric_cols = {"trace_id", "inputs", "outputs", "trace", "span_id", "assessments"}

    for profile_name, eval_result in results.items():
        if eval_result is None or eval_result.result_df is None or eval_result.result_df.empty:
            continue

        result_df = eval_result.result_df

        for _, row in result_df.iterrows():
            trace_id = row.get("trace_id")

            inputs = row.get("inputs") or ""
            outputs = row.get("outputs") or ""
            inputs_str = json.dumps(inputs) if isinstance(
                inputs, dict) else str(inputs)
            outputs_str = json.dumps(outputs) if isinstance(
                outputs, dict) else str(outputs)

            # Build rationale lookup from assessments
            assessments = row.get("assessments") or []
            rationale_map = {}
            for assessment in assessments:
                name = assessment.get("assessment_name")
                if not name:
                    continue
                # If multiple assessments for same metric, keep latest by last_update_time
                existing = rationale_map.get(name)
                if existing is None or assessment.get("last_update_time", "") > existing.get("last_update_time", ""):
                    rationale_map[name] = assessment

            for col in result_df.columns:
                if col in non_metric_cols:
                    continue

                metric_value = row.get(col)
                if pd.isna(metric_value):
                    continue

                # Get rationale from assessments
                assessment = rationale_map.get(col, {})
                rationale = assessment.get("rationale", "")

                records.append({
                    "experiment_id": experiment_id,
                    "trace_id": trace_id,
                    "request": inputs_str,
                    "response": outputs_str,
                    "profile": profile_name,
                    "metric_name": col,
                    "metric_value": str(metric_value),
                    "rationale": rationale,
                    "eval_timestamp": eval_timestamp,
                })

    if not records:
        print("No records to export")
        return

    spark = SparkSession.builder.getOrCreate()
    full_table_name = f"{catalog}.{schema}.{table_name}"
    source_df = spark.createDataFrame(pd.DataFrame(records))

    try:
        spark.sql(f"DESCRIBE TABLE {full_table_name}")
        delta_table = DeltaTable.forName(spark, full_table_name)
        delta_table.alias("target").merge(
            source_df.alias("source"),
            "target.trace_id = source.trace_id AND target.metric_name = source.metric_name"
        ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
        print(f"Export successful: {len(records)} records merged to {full_table_name}")
    except Exception as e:
        if "Table or view not found" in str(e) or "DESCRIBE TABLE" in str(e):
            source_df.write.format("delta").mode(
                "overwrite").saveAsTable(full_table_name)
            print(f"Export successful: {len(records)} records written to new table {full_table_name}")
        else:
            print(f"Export failed: {e}")
            raise
