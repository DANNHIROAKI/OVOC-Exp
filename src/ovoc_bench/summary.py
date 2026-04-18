from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .utils import ensure_dir, write_text


def _fmt_pct(x: float | int | None) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{float(x):.2f}"

def _fmt_int(x: float | int | None) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{int(round(float(x)))}"

def _fmt_sec_from_ms(x: float | int | None) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{float(x)/1000:.2f}s"


def aggregate_sample_level(
    task_direct_df: pd.DataFrame,
    task_amortized_df: pd.DataFrame,
) -> pd.DataFrame:
    direct = (
        task_direct_df.groupby(["rerun_id", "group_id", "sample_id"], as_index=False)
        .agg(
            correct_cases=("judge_correct", "sum"),
            case_count=("case_uid", "count"),
            qa_elapsed_sum_ms=("qa_elapsed_ms", "sum"),
            qa_elapsed_mean_ms=("qa_elapsed_ms", "mean"),
            qa_elapsed_p90_ms=("qa_elapsed_ms", lambda s: float(np.percentile(s, 90))),
            qa_retry_rate=("qa_retry_count", lambda s: float((pd.Series(s) > 0).mean())),
            qa_error_rate=("qa_error_flag", "mean"),
        )
    )
    amort = (
        task_amortized_df.groupby(["rerun_id", "group_id", "sample_id"], as_index=False)
        .agg(
            input_tokens_total=("task_input_tokens_amortized", "sum"),
            total_tokens_total=("task_total_tokens_amortized", "sum"),
            elapsed_total_ms=("task_elapsed_ms_amortized", "sum"),
            amortized_elapsed_mean_ms=("task_elapsed_ms_amortized", "mean"),
        )
    )
    merged = direct.merge(amort, on=["rerun_id", "group_id", "sample_id"], how="inner")
    merged["completion_rate"] = merged["correct_cases"] / merged["case_count"]
    merged["cost_per_correct"] = np.where(
        merged["correct_cases"] > 0,
        merged["input_tokens_total"] / merged["correct_cases"],
        np.nan,
    )
    return merged


def aggregate_group_level(
    task_direct_df: pd.DataFrame,
    task_amortized_df: pd.DataFrame,
) -> pd.DataFrame:
    sample_level = aggregate_sample_level(task_direct_df, task_amortized_df)
    group_df = (
        sample_level.groupby(["rerun_id", "group_id"], as_index=False)
        .agg(
            correct_cases=("correct_cases", "sum"),
            case_count=("case_count", "sum"),
            input_tokens_total=("input_tokens_total", "sum"),
            total_tokens_total=("total_tokens_total", "sum"),
            elapsed_total_ms=("elapsed_total_ms", "sum"),
            qa_elapsed_sum_ms=("qa_elapsed_sum_ms", "sum"),
            qa_retry_rate=("qa_retry_rate", "mean"),
            qa_error_rate=("qa_error_rate", "mean"),
        )
    )
    group_df["task_completion_rate"] = group_df["correct_cases"] / group_df["case_count"]
    group_df["direct_qa_mean_elapsed_ms"] = group_df["qa_elapsed_sum_ms"] / group_df["case_count"]
    group_df["cost_per_correct"] = np.where(
        group_df["correct_cases"] > 0,
        group_df["input_tokens_total"] / group_df["correct_cases"],
        np.nan,
    )
    group_df["elapsed_per_correct_ms"] = np.where(
        group_df["correct_cases"] > 0,
        group_df["elapsed_total_ms"] / group_df["correct_cases"],
        np.nan,
    )
    return group_df


def aggregate_category_level(task_amortized_df: pd.DataFrame) -> pd.DataFrame:
    cat = (
        task_amortized_df.groupby(["rerun_id", "group_id", "category"], as_index=False)
        .agg(
            correct_cases=("judge_correct", "sum"),
            case_count=("case_uid", "count"),
            avg_input_tokens_amortized=("task_input_tokens_amortized", "mean"),
            avg_elapsed_ms_amortized=("task_elapsed_ms_amortized", "mean"),
        )
    )
    cat["completion_rate"] = cat["correct_cases"] / cat["case_count"]
    return cat


def bootstrap_pairwise_ci(
    sample_level_df: pd.DataFrame,
    *,
    rerun_id: str,
    left_group: str,
    right_group: str,
    n_boot: int = 10000,
    seed: int = 42,
) -> dict[str, tuple[float, float, float]]:
    left = sample_level_df[(sample_level_df["rerun_id"] == rerun_id) & (sample_level_df["group_id"] == left_group)].copy()
    right = sample_level_df[(sample_level_df["rerun_id"] == rerun_id) & (sample_level_df["group_id"] == right_group)].copy()
    merged = left.merge(
        right,
        on=["rerun_id", "sample_id"],
        suffixes=("_l", "_r"),
        how="inner",
    )
    if merged.empty:
        return {}
    rng = np.random.default_rng(seed)
    n = len(merged)

    def _bootstrap(stat_fn: Callable[[pd.DataFrame], float]) -> tuple[float, float, float]:
        obs = float(stat_fn(merged))
        samples = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boot = merged.iloc[idx]
            samples.append(float(stat_fn(boot)))
        lo, hi = np.percentile(samples, [2.5, 97.5])
        return obs, float(lo), float(hi)

    stats = {
        "completion_rate_diff_pp": _bootstrap(
            lambda d: 100.0 * (
                d["correct_cases_l"].sum() / d["case_count_l"].sum()
                - d["correct_cases_r"].sum() / d["case_count_r"].sum()
            )
        ),
        "input_tokens_diff": _bootstrap(
            lambda d: d["input_tokens_total_l"].sum() - d["input_tokens_total_r"].sum()
        ),
        "elapsed_total_diff_ms": _bootstrap(
            lambda d: d["elapsed_total_ms_l"].sum() - d["elapsed_total_ms_r"].sum()
        ),
        "cost_per_correct_diff": _bootstrap(
            lambda d: (
                d["input_tokens_total_l"].sum() / d["correct_cases_l"].sum()
                - d["input_tokens_total_r"].sum() / d["correct_cases_r"].sum()
            )
        ),
        "direct_qa_mean_elapsed_diff_ms": _bootstrap(
            lambda d: (
                d["qa_elapsed_sum_ms_l"].sum() / d["case_count_l"].sum()
                - d["qa_elapsed_sum_ms_r"].sum() / d["case_count_r"].sum()
            )
        ),
        "amortized_elapsed_mean_diff_ms": _bootstrap(
            lambda d: (
                d["elapsed_total_ms_l"].sum() / d["case_count_l"].sum()
                - d["elapsed_total_ms_r"].sum() / d["case_count_r"].sum()
            )
        ),
    }
    return stats


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._\n"
    return df.to_markdown(index=False) + "\n"


def generate_summary_markdown(
    cfg: ExperimentConfig,
    *,
    task_direct_df: pd.DataFrame,
    task_amortized_df: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    output_dir = ensure_dir(output_dir)
    group_map = cfg.group_map()
    sample_level = aggregate_sample_level(task_direct_df, task_amortized_df)
    group_level = aggregate_group_level(task_direct_df, task_amortized_df)
    category_level = aggregate_category_level(task_amortized_df)

    # main table
    group_level = group_level.copy()
    group_level["group_label"] = group_level["group_id"].map(lambda x: group_map[x].label)
    group_level["plugins.slots.memory"] = group_level["group_id"].map(lambda x: group_map[x].memory_slot)
    group_level["plugins.slots.contextEngine"] = group_level["group_id"].map(lambda x: group_map[x].context_engine)
    group_level["plugins.entries.openviking.enabled"] = group_level["group_id"].map(lambda x: group_map[x].openviking_enabled)
    group_level["plugins.deny"] = group_level["group_id"].map(lambda x: str(group_map[x].deny))
    main_rows = group_level[[
        "rerun_id",
        "group_id",
        "group_label",
        "plugins.slots.memory",
        "plugins.slots.contextEngine",
        "plugins.entries.openviking.enabled",
        "plugins.deny",
        "task_completion_rate",
        "input_tokens_total",
        "elapsed_total_ms",
        "direct_qa_mean_elapsed_ms",
    ]].copy()
    main_rows["task_completion_rate"] = (main_rows["task_completion_rate"] * 100).map(_fmt_pct)
    main_rows["input_tokens_total"] = main_rows["input_tokens_total"].map(_fmt_int)
    main_rows["elapsed_total_ms"] = main_rows["elapsed_total_ms"].map(_fmt_sec_from_ms)
    main_rows["direct_qa_mean_elapsed_ms"] = main_rows["direct_qa_mean_elapsed_ms"].map(_fmt_sec_from_ms)
    main_path = output_dir / "main_table.md"
    write_text(main_path, "# Main Table\n\n" + _markdown_table(main_rows))

    # planned comparisons
    comparison_specs = [
        ("C1", "G1", "G2", "部署比较"),
        ("C2", "G3", "G2", "单因素核心比较"),
        ("C3", "G3", "G1", "OV 内部 memory-core 比较"),
    ]
    rows = []
    for rerun_id in sorted(group_level["rerun_id"].unique()):
        sample_r = sample_level[sample_level["rerun_id"] == rerun_id]
        group_r = group_level[group_level["rerun_id"] == rerun_id].set_index("group_id")
        for cid, left, right, kind in comparison_specs:
            if left not in group_r.index or right not in group_r.index:
                continue
            l = group_r.loc[left]
            r = group_r.loc[right]
            ci = bootstrap_pairwise_ci(sample_level, rerun_id=rerun_id, left_group=left, right_group=right)
            comp_rate_diff = (l["task_completion_rate"] - r["task_completion_rate"]) * 100
            rel_lift = ((l["task_completion_rate"] - r["task_completion_rate"]) / r["task_completion_rate"] * 100) if r["task_completion_rate"] else np.nan
            input_diff = l["input_tokens_total"] - r["input_tokens_total"]
            input_reduction = ((r["input_tokens_total"] - l["input_tokens_total"]) / r["input_tokens_total"] * 100) if r["input_tokens_total"] else np.nan
            elapsed_diff = l["elapsed_total_ms"] - r["elapsed_total_ms"]
            direct_latency_diff = l["direct_qa_mean_elapsed_ms"] - r["direct_qa_mean_elapsed_ms"]
            cpp_diff = l["cost_per_correct"] - r["cost_per_correct"]
            rows.append({
                "rerun_id": rerun_id,
                "比较": f"{cid}: {left} vs {right}",
                "比较性质": kind,
                "完成率差值（pp）": f"{comp_rate_diff:.2f}",
                "完成率差值95%CI": (
                    f"[{ci['completion_rate_diff_pp'][1]:.2f}, {ci['completion_rate_diff_pp'][2]:.2f}]"
                    if "completion_rate_diff_pp" in ci else ""
                ),
                "相对提升（%）": f"{rel_lift:.2f}" if pd.notna(rel_lift) else "",
                "输入 token 差值": _fmt_int(input_diff),
                "输入 token 差值95%CI": (
                    f"[{ci['input_tokens_diff'][1]:.0f}, {ci['input_tokens_diff'][2]:.0f}]"
                    if "input_tokens_diff" in ci else ""
                ),
                "输入 token 降幅（%）": f"{input_reduction:.2f}" if pd.notna(input_reduction) else "",
                "总耗时差值": _fmt_sec_from_ms(elapsed_diff),
                "总耗时差值95%CI": (
                    f"[{ci['elapsed_total_diff_ms'][1]/1000:.2f}s, {ci['elapsed_total_diff_ms'][2]/1000:.2f}s]"
                    if "elapsed_total_diff_ms" in ci else ""
                ),
                "直接 QA 平均耗时差值": _fmt_sec_from_ms(direct_latency_diff),
                "每正确题成本变化": f"{cpp_diff:.2f}" if pd.notna(cpp_diff) else "",
            })
    comparisons_df = pd.DataFrame(rows)
    planned_path = output_dir / "planned_comparisons.md"
    write_text(planned_path, "# Planned Comparisons\n\n" + _markdown_table(comparisons_df))

    # sample breakdown
    sample_break = sample_level.copy()
    sample_break["completion_rate"] = (sample_break["completion_rate"] * 100).map(_fmt_pct)
    sample_break["input_tokens_total"] = sample_break["input_tokens_total"].map(_fmt_int)
    sample_break["elapsed_total_ms"] = sample_break["elapsed_total_ms"].map(_fmt_sec_from_ms)
    sample_break = sample_break.rename(columns={
        "completion_rate": "完成率(%)",
        "input_tokens_total": "输入 token 总计",
        "elapsed_total_ms": "总耗时",
    })
    sample_path = output_dir / "sample_breakdown.md"
    write_text(sample_path, "# Sample Breakdown\n\n" + _markdown_table(sample_break))

    # category breakdown
    cat_break = category_level.copy()
    cat_break["completion_rate"] = (cat_break["completion_rate"] * 100).map(_fmt_pct)
    cat_break["avg_input_tokens_amortized"] = cat_break["avg_input_tokens_amortized"].map(lambda x: f"{x:.2f}")
    cat_break["avg_elapsed_ms_amortized"] = cat_break["avg_elapsed_ms_amortized"].map(_fmt_sec_from_ms)
    category_path = output_dir / "category_breakdown.md"
    write_text(category_path, "# Category Breakdown\n\n" + _markdown_table(cat_break))

    # latency breakdown
    latency_rows = (
        task_amortized_df.groupby(["rerun_id", "group_id"], as_index=False)
        .agg(
            qa_elapsed_mean_ms=("qa_elapsed_ms", "mean"),
            qa_elapsed_median_ms=("qa_elapsed_ms", "median"),
            qa_elapsed_p90_ms=("qa_elapsed_ms", lambda s: float(np.percentile(s, 90))),
            amortized_elapsed_mean_ms=("task_elapsed_ms_amortized", "mean"),
            amortized_elapsed_median_ms=("task_elapsed_ms_amortized", "median"),
            amortized_elapsed_p90_ms=("task_elapsed_ms_amortized", lambda s: float(np.percentile(s, 90))),
        )
    )
    for col in latency_rows.columns:
        if col.endswith("_ms"):
            latency_rows[col] = latency_rows[col].map(_fmt_sec_from_ms)
    latency_path = output_dir / "latency_breakdown.md"
    write_text(latency_path, "# Latency Breakdown\n\n" + _markdown_table(latency_rows))

    schema_lines = [
        "# Per-task Schema",
        "",
        "## task_metrics_direct",
        "",
        *[f"- `{col}`" for col in task_direct_df.columns.tolist()],
        "",
        "## task_metrics_amortized",
        "",
        *[f"- `{col}`" for col in task_amortized_df.columns.tolist()],
        "",
    ]
    schema_path = output_dir / "per_task_schema.md"
    write_text(schema_path, "\n".join(schema_lines))
    return {
        "main_table": main_path,
        "planned_comparisons": planned_path,
        "sample_breakdown": sample_path,
        "category_breakdown": category_path,
        "latency_breakdown": latency_path,
        "per_task_schema": schema_path,
    }
