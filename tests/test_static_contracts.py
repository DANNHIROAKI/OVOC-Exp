from __future__ import annotations

import unittest
from pathlib import Path

from ovoc_bench.config import load_experiment_config
from ovoc_bench.dataset import load_dataset, validate_dataset
from ovoc_bench.metrics import (
    amortize_tasks,
    build_sample_ingest_record,
    build_task_direct_record,
    reconcile_group_totals,
    to_dataframe,
)
from ovoc_bench.openviking import OvUsage
from ovoc_bench.runner import load_runner


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "experiment.yaml"


class StaticContractsTest(unittest.TestCase):
    def test_dataset_validation_and_group_lock(self) -> None:
        cfg = load_experiment_config(CONFIG_PATH)
        self.assertEqual([g.id for g in cfg.groups], ["G1", "G2", "G3"])
        self.assertEqual(cfg.versions.openclaw, "2026.4.14")
        self.assertEqual(cfg.versions.openviking, "0.3.8")
        self.assertEqual([g.slug for g in cfg.groups], ["ov-no-memory", "no-ov-stock", "ov-stock"])

        bundle = load_dataset(cfg.dataset, tail=cfg.runtime.ingest_tail)
        validation = validate_dataset(bundle, cfg.dataset)
        self.assertTrue(validation["ok"])
        self.assertEqual(validation["total_cases"], 1540)
        self.assertEqual(validation["sample_count"], 10)
        self.assertEqual(validation["category_5_count"], 0)

    def test_rotation_contract(self) -> None:
        runner = load_runner(CONFIG_PATH)
        runner._dataset = load_dataset(runner.cfg.dataset, tail=runner.cfg.runtime.ingest_tail)
        blocks = runner.plan_blocks()
        self.assertGreaterEqual(len(blocks), 9)

        chunk1 = blocks[0:3]
        chunk2 = blocks[3:6]
        chunk3 = blocks[6:9]

        self.assertEqual({b.sample.sample_id for b in chunk1}, {chunk1[0].sample.sample_id})
        self.assertEqual({b.sample.sample_id for b in chunk2}, {chunk2[0].sample.sample_id})
        self.assertEqual({b.sample.sample_id for b in chunk3}, {chunk3[0].sample.sample_id})

        self.assertEqual([b.group.id for b in chunk1], ["G1", "G2", "G3"])
        self.assertEqual([b.group.id for b in chunk2], ["G2", "G3", "G1"])
        self.assertEqual([b.group.id for b in chunk3], ["G3", "G1", "G2"])

    def test_amortization_reconciles(self) -> None:
        sample_record = build_sample_ingest_record(
            run_id="run",
            group_id="G1",
            rerun_id="R1",
            sample_id="conv-x",
            sessions_ingested=2,
            ingest_start_ts="t0",
            ingest_end_ts="t1",
            ingest_elapsed_ms=1200,
            gateway_usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            ov_usage=OvUsage(input_tokens=20, output_tokens=7, total_tokens=27),
            ov_barrier_wait_ms=300,
            post_reset_quiet_wait_ms=0,
            barrier_session_id="sid",
            barrier_detail={"commit_count": 1, "memories_extracted": {"total": 2}},
            barrier_context={"latest_archive_overview": "ready"},
        )
        direct = [
            build_task_direct_record(
                run_id="run",
                group_id="G1",
                rerun_id="R1",
                sample_id="conv-x",
                case_uid=f"case-{i}",
                case_id=i,
                sample_idx=1,
                qa_idx_within_sample=i,
                category=1,
                question="q",
                gold_answer="g",
                prediction="p",
                judge_correct=(i % 2 == 0),
                judge_reasoning_raw="ok",
                judge_model_id="judge-model",
                judge_prompt_version="judge_prompt_v1",
                qa_start_ts="t0",
                qa_end_ts="t1",
                qa_elapsed_ms=100 + i,
                qa_retry_count=0,
                qa_error_flag=False,
                gateway_usage={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
                ov_usage=OvUsage(input_tokens=4, output_tokens=1, total_tokens=5),
                qa_session_id="sid",
            )
            for i in range(1, 5)
        ]
        amort = amortize_tasks(direct, sample_record)
        rec = reconcile_group_totals(to_dataframe([sample_record]), to_dataframe(amort))
        self.assertTrue(rec["ok"])


if __name__ == "__main__":
    unittest.main()
