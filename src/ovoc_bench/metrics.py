from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .openviking import OvUsage


SAMPLE_INGEST_COLUMNS = [
    "run_id",
    "group_id",
    "rerun_id",
    "sample_id",
    "sessions_ingested",
    "ingest_start_ts",
    "ingest_end_ts",
    "ingest_elapsed_ms",
    "ingest_gateway_input_tokens",
    "ingest_gateway_output_tokens",
    "ingest_gateway_total_tokens",
    "ingest_ov_internal_input_tokens",
    "ingest_ov_internal_output_tokens",
    "ingest_ov_internal_total_tokens",
    "ingest_input_tokens_total",
    "ingest_total_tokens_total",
    "ov_barrier_wait_ms",
    "post_reset_quiet_wait_ms",
    "ov_usage_source",
    "ov_usage_matched_events",
    "barrier_session_id",
    "barrier_commit_count",
    "barrier_memories_extracted_total",
    "barrier_archive_overview_present",
]

TASK_DIRECT_COLUMNS = [
    "run_id",
    "group_id",
    "rerun_id",
    "sample_id",
    "case_uid",
    "case_id",
    "sample_idx",
    "qa_idx_within_sample",
    "category",
    "question",
    "gold_answer",
    "prediction",
    "judge_correct",
    "judge_reasoning_raw",
    "judge_model_id",
    "judge_prompt_version",
    "qa_start_ts",
    "qa_end_ts",
    "qa_elapsed_ms",
    "qa_retry_count",
    "qa_error_flag",
    "qa_gateway_input_tokens",
    "qa_gateway_output_tokens",
    "qa_gateway_total_tokens",
    "qa_ov_internal_input_tokens",
    "qa_ov_internal_output_tokens",
    "qa_ov_internal_total_tokens",
    "qa_input_tokens_direct",
    "qa_output_tokens_direct",
    "qa_total_tokens_direct",
    "qa_ov_usage_source",
    "qa_ov_usage_matched_events",
    "qa_session_id",
]

TASK_AMORTIZED_COLUMNS = TASK_DIRECT_COLUMNS + [
    "alloc_ingest_input_tokens",
    "alloc_ingest_output_tokens",
    "alloc_ingest_total_tokens",
    "alloc_ingest_elapsed_ms",
    "task_input_tokens_amortized",
    "task_output_tokens_amortized",
    "task_total_tokens_amortized",
    "task_elapsed_ms_amortized",
]


def build_sample_ingest_record(
    *,
    run_id: str,
    group_id: str,
    rerun_id: str,
    sample_id: str,
    sessions_ingested: int,
    ingest_start_ts: str,
    ingest_end_ts: str,
    ingest_elapsed_ms: int,
    gateway_usage: dict[str, int],
    ov_usage: OvUsage,
    ov_barrier_wait_ms: int,
    post_reset_quiet_wait_ms: int,
    barrier_session_id: str | None,
    barrier_detail: dict[str, Any] | None,
    barrier_context: dict[str, Any] | None,
) -> dict[str, Any]:
    record = {
        "run_id": run_id,
        "group_id": group_id,
        "rerun_id": rerun_id,
        "sample_id": sample_id,
        "sessions_ingested": sessions_ingested,
        "ingest_start_ts": ingest_start_ts,
        "ingest_end_ts": ingest_end_ts,
        "ingest_elapsed_ms": int(ingest_elapsed_ms),
        "ingest_gateway_input_tokens": int(gateway_usage.get("input_tokens", 0)),
        "ingest_gateway_output_tokens": int(gateway_usage.get("output_tokens", 0)),
        "ingest_gateway_total_tokens": int(gateway_usage.get("total_tokens", 0)),
        "ingest_ov_internal_input_tokens": int(ov_usage.input_tokens),
        "ingest_ov_internal_output_tokens": int(ov_usage.output_tokens),
        "ingest_ov_internal_total_tokens": int(ov_usage.total_tokens),
        "ingest_input_tokens_total": int(gateway_usage.get("input_tokens", 0) + ov_usage.input_tokens),
        "ingest_total_tokens_total": int(gateway_usage.get("total_tokens", 0) + ov_usage.total_tokens),
        "ov_barrier_wait_ms": int(ov_barrier_wait_ms),
        "post_reset_quiet_wait_ms": int(post_reset_quiet_wait_ms),
        "ov_usage_source": ov_usage.source,
        "ov_usage_matched_events": int(ov_usage.matched_events),
        "barrier_session_id": barrier_session_id,
        "barrier_commit_count": int((barrier_detail or {}).get("commit_count", 0) or 0),
        "barrier_memories_extracted_total": int(((barrier_detail or {}).get("memories_extracted") or {}).get("total", 0) or 0),
        "barrier_archive_overview_present": bool(((barrier_context or {}).get("latest_archive_overview") or "").strip()),
    }
    return record


def build_task_direct_record(
    *,
    run_id: str,
    group_id: str,
    rerun_id: str,
    sample_id: str,
    case_uid: str,
    case_id: int,
    sample_idx: int,
    qa_idx_within_sample: int,
    category: int,
    question: str,
    gold_answer: str,
    prediction: str,
    judge_correct: bool,
    judge_reasoning_raw: str,
    judge_model_id: str,
    judge_prompt_version: str,
    qa_start_ts: str,
    qa_end_ts: str,
    qa_elapsed_ms: int,
    qa_retry_count: int,
    qa_error_flag: bool,
    gateway_usage: dict[str, int],
    ov_usage: OvUsage,
    qa_session_id: str | None,
) -> dict[str, Any]:
    qa_gateway_input = int(gateway_usage.get("input_tokens", 0))
    qa_gateway_output = int(gateway_usage.get("output_tokens", 0))
    qa_gateway_total = int(gateway_usage.get("total_tokens", 0))
    record = {
        "run_id": run_id,
        "group_id": group_id,
        "rerun_id": rerun_id,
        "sample_id": sample_id,
        "case_uid": case_uid,
        "case_id": int(case_id),
        "sample_idx": int(sample_idx),
        "qa_idx_within_sample": int(qa_idx_within_sample),
        "category": int(category),
        "question": question,
        "gold_answer": gold_answer,
        "prediction": prediction,
        "judge_correct": bool(judge_correct),
        "judge_reasoning_raw": judge_reasoning_raw,
        "judge_model_id": judge_model_id,
        "judge_prompt_version": judge_prompt_version,
        "qa_start_ts": qa_start_ts,
        "qa_end_ts": qa_end_ts,
        "qa_elapsed_ms": int(qa_elapsed_ms),
        "qa_retry_count": int(qa_retry_count),
        "qa_error_flag": bool(qa_error_flag),
        "qa_gateway_input_tokens": qa_gateway_input,
        "qa_gateway_output_tokens": qa_gateway_output,
        "qa_gateway_total_tokens": qa_gateway_total,
        "qa_ov_internal_input_tokens": int(ov_usage.input_tokens),
        "qa_ov_internal_output_tokens": int(ov_usage.output_tokens),
        "qa_ov_internal_total_tokens": int(ov_usage.total_tokens),
        "qa_input_tokens_direct": int(qa_gateway_input + ov_usage.input_tokens),
        "qa_output_tokens_direct": int(qa_gateway_output + ov_usage.output_tokens),
        "qa_total_tokens_direct": int(qa_gateway_total + ov_usage.total_tokens),
        "qa_ov_usage_source": ov_usage.source,
        "qa_ov_usage_matched_events": int(ov_usage.matched_events),
        "qa_session_id": qa_session_id,
    }
    return record


def amortize_tasks(
    direct_records: list[dict[str, Any]],
    sample_ingest_record: dict[str, Any],
) -> list[dict[str, Any]]:
    n = len(direct_records)
    if n <= 0:
        return []
    alloc_in = sample_ingest_record["ingest_input_tokens_total"] / n
    alloc_out = (
        sample_ingest_record["ingest_gateway_output_tokens"]
        + sample_ingest_record["ingest_ov_internal_output_tokens"]
    ) / n
    alloc_total = sample_ingest_record["ingest_total_tokens_total"] / n
    alloc_elapsed = sample_ingest_record["ingest_elapsed_ms"] / n
    rows: list[dict[str, Any]] = []
    for row in direct_records:
        item = dict(row)
        item["alloc_ingest_input_tokens"] = alloc_in
        item["alloc_ingest_output_tokens"] = alloc_out
        item["alloc_ingest_total_tokens"] = alloc_total
        item["alloc_ingest_elapsed_ms"] = alloc_elapsed
        item["task_input_tokens_amortized"] = row["qa_input_tokens_direct"] + alloc_in
        item["task_output_tokens_amortized"] = row["qa_output_tokens_direct"] + alloc_out
        item["task_total_tokens_amortized"] = row["qa_total_tokens_direct"] + alloc_total
        item["task_elapsed_ms_amortized"] = row["qa_elapsed_ms"] + alloc_elapsed
        rows.append(item)
    return rows


def to_dataframe(records: list[dict[str, Any]], columns: list[str] | None = None) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if columns:
        for col in columns:
            if col not in df.columns:
                df[col] = pd.NA
        df = df[columns]
    return df


def reconcile_group_totals(
    sample_ingest_df: pd.DataFrame,
    task_amortized_df: pd.DataFrame,
    *,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    expected_input = float(sample_ingest_df["ingest_input_tokens_total"].sum() + task_amortized_df["qa_input_tokens_direct"].sum())
    expected_total = float(sample_ingest_df["ingest_total_tokens_total"].sum() + task_amortized_df["qa_total_tokens_direct"].sum())
    expected_elapsed = float(sample_ingest_df["ingest_elapsed_ms"].sum() + task_amortized_df["qa_elapsed_ms"].sum())

    actual_input = float(task_amortized_df["task_input_tokens_amortized"].sum())
    actual_total = float(task_amortized_df["task_total_tokens_amortized"].sum())
    actual_elapsed = float(task_amortized_df["task_elapsed_ms_amortized"].sum())

    ok = (
        abs(expected_input - actual_input) <= tolerance
        and abs(expected_total - actual_total) <= tolerance
        and abs(expected_elapsed - actual_elapsed) <= tolerance
    )
    return {
        "ok": ok,
        "expected_input": expected_input,
        "actual_input": actual_input,
        "expected_total": expected_total,
        "actual_total": actual_total,
        "expected_elapsed": expected_elapsed,
        "actual_elapsed": actual_elapsed,
    }
