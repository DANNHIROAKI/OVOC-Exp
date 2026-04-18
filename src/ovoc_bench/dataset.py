from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import pandas as pd

from .config import DatasetConfig
from .utils import (
    load_json,
    load_jsonl,
    normalize_answer_text,
    sha256_file,
    stable_json_sha1,
)


@dataclass(slots=True)
class SampleSession:
    session_key: str
    date_time: str
    message: str


@dataclass(slots=True)
class CaseRecord:
    case_uid: str
    case_id: int
    sample_idx: int
    sample_id: str
    qa_idx_within_sample: int
    category: int
    question: str
    gold_answer: str
    evidence: list[Any]


@dataclass(slots=True)
class SampleRecord:
    sample_idx: int
    sample_id: str
    raw: dict[str, Any]
    sessions: list[SampleSession]
    cases: list[CaseRecord]


@dataclass(slots=True)
class DatasetBundle:
    json_path: Path
    jsonl_path: Path
    manifest_path: Path
    raw_samples: list[dict[str, Any]]
    samples: list[SampleRecord]
    cases_df: pd.DataFrame
    manifest: dict[str, Any]


def format_locomo_message(msg: dict[str, Any]) -> str:
    speaker = msg.get("speaker", "unknown")
    text = msg.get("text", "")
    line = f"{speaker}: {text}"
    img_urls = msg.get("img_url", [])
    if isinstance(img_urls, str):
        img_urls = [img_urls]
    blip = msg.get("blip_caption", "")
    if img_urls:
        for url in img_urls:
            caption = f": {blip}" if blip else ""
            line += f"\n{url}{caption}"
    elif blip:
        line += f"\n({blip})"
    return line


def build_session_messages(sample: dict[str, Any], tail: str) -> list[SampleSession]:
    conv = sample["conversation"]
    session_keys = sorted(
        [k for k in conv if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda k: int(k.split("_")[1]),
    )
    out: list[SampleSession] = []
    for sk in session_keys:
        dt_key = f"{sk}_date_time"
        date_time = conv.get(dt_key, "")
        parts = [f"[group chat conversation: {date_time}]"]
        for msg in conv[sk]:
            parts.append(format_locomo_message(msg))
        if tail:
            parts.append(tail)
        out.append(
            SampleSession(
                session_key=sk,
                date_time=date_time,
                message="\n\n".join(parts),
            )
        )
    return out


def _case_uid_from_row(row: dict[str, Any]) -> str:
    if "case_uid" in row and str(row["case_uid"]).strip():
        return str(row["case_uid"])
    if "case_id" in row:
        return f"case-{int(row['case_id']):06d}"
    fingerprint = {
        "sample_id": row["sample_id"],
        "qa_idx_within_sample": row.get("qa_idx_within_sample"),
        "category": row.get("category"),
        "question": row["question"],
        "answer": normalize_answer_text(row["answer"]),
    }
    return f"case-{stable_json_sha1(fingerprint)[:16]}"


def load_dataset(cfg: DatasetConfig, tail: str) -> DatasetBundle:
    raw_samples = load_json(cfg.vendor_json)
    manifest = load_json(cfg.vendor_manifest)
    flat_rows = load_jsonl(cfg.vendor_jsonl)
    if not isinstance(raw_samples, list):
        raise ValueError("Vendor JSON dataset must be a list of samples.")
    if not isinstance(flat_rows, list):
        raise ValueError("Vendor JSONL dataset must decode to rows.")

    case_rows: list[dict[str, Any]] = []
    for row in flat_rows:
        row = dict(row)
        row["case_uid"] = _case_uid_from_row(row)
        row["gold_answer"] = normalize_answer_text(row.get("answer"))
        case_rows.append(row)

    cases_df = pd.DataFrame(case_rows)
    samples: list[SampleRecord] = []
    grouped = cases_df.groupby("sample_id", sort=False)

    for sample_idx, raw_sample in enumerate(raw_samples):
        sample_id = raw_sample["sample_id"]
        if sample_id not in grouped.groups:
            raise ValueError(f"Sample {sample_id} missing from flat case index.")
        sample_cases_df = grouped.get_group(sample_id).sort_values(["qa_idx_within_sample", "case_id"])
        sessions = build_session_messages(raw_sample, tail=tail)
        cases: list[CaseRecord] = []
        for row in sample_cases_df.to_dict(orient="records"):
            cases.append(
                CaseRecord(
                    case_uid=str(row["case_uid"]),
                    case_id=int(row["case_id"]),
                    sample_idx=int(row["sample_idx"]),
                    sample_id=str(row["sample_id"]),
                    qa_idx_within_sample=int(row["qa_idx_within_sample"]),
                    category=int(row["category"]),
                    question=str(row["question"]),
                    gold_answer=str(row["gold_answer"]),
                    evidence=list(row.get("evidence", [])),
                )
            )
        samples.append(
            SampleRecord(
                sample_idx=sample_idx,
                sample_id=sample_id,
                raw=raw_sample,
                sessions=sessions,
                cases=cases,
            )
        )
    return DatasetBundle(
        json_path=cfg.vendor_json,
        jsonl_path=cfg.vendor_jsonl,
        manifest_path=cfg.vendor_manifest,
        raw_samples=raw_samples,
        samples=samples,
        cases_df=cases_df,
        manifest=manifest,
    )


def validate_dataset(bundle: DatasetBundle, cfg: DatasetConfig) -> dict[str, Any]:
    df = bundle.cases_df.copy()
    total_cases = int(len(df))
    category_5_count = int((df["category"].astype(int) == int(cfg.expected_removed_category)).sum())
    missing_sample_id = int(df["sample_id"].isna().sum())
    duplicate_case_uid = int(df["case_uid"].duplicated().sum())
    sample_counts = (
        df.groupby("sample_id", sort=False)
        .size()
        .rename("case_count")
        .reset_index()
        .to_dict(orient="records")
    )
    sha_summary = {
        "json_sha256": sha256_file(bundle.json_path),
        "jsonl_sha256": sha256_file(bundle.jsonl_path),
    }
    ok = (
        total_cases == cfg.expected_total_cases
        and len(bundle.samples) == cfg.expected_sample_count
        and category_5_count == 0
        and missing_sample_id == 0
        and duplicate_case_uid == 0
    )
    return {
        "ok": ok,
        "total_cases": total_cases,
        "expected_total_cases": cfg.expected_total_cases,
        "sample_count": len(bundle.samples),
        "expected_sample_count": cfg.expected_sample_count,
        "category_5_count": category_5_count,
        "missing_sample_id": missing_sample_id,
        "duplicate_case_uid": duplicate_case_uid,
        "sample_counts": sample_counts,
        "sha256": sha_summary,
        "manifest_summary": bundle.manifest.get("stats", {}),
    }
