from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import os
import platform
import traceback
import time

import pandas as pd

from .config import ExperimentConfig, GroupSpec, load_experiment_config
from .dataset import DatasetBundle, SampleRecord, load_dataset, validate_dataset
from .gitmeta import clone_or_update, get_commit_sha, get_remote_url
from .judge import JudgeConfig, judge_records
from .metrics import (
    SAMPLE_INGEST_COLUMNS,
    TASK_AMORTIZED_COLUMNS,
    TASK_DIRECT_COLUMNS,
    amortize_tasks,
    build_sample_ingest_record,
    build_task_direct_record,
    reconcile_group_totals,
    to_dataframe,
)
from .openclaw import (
    apply_group_config,
    archive_session_file,
    assert_group_runtime,
    config_get,
    config_set,
    config_validate,
    get_openclaw_version,
    get_session_id_for_user,
    inspect_plugin,
    install_openclaw,
    install_plugin,
    load_openclaw_config,
    onboard_custom_provider,
    openclaw_bin,
    post_response,
    read_runtime_config_summary,
    save_redacted_openclaw_config,
    start_gateway,
    wait_for_gateway_ready,
    write_openclaw_dotenv,
)
from .openviking import (
    OpenVikingInspector,
    OvUsage,
    aggregate_usage_from_events,
    extract_memory_total,
    get_openviking_version,
    install_openviking,
    merge_usage,
    openviking_python_bin,
    parse_ov_log,
    render_ov_conf,
    usage_from_payload,
    wait_for_commit_visibility,
)
from .subproc import run_cmd
from .summary import generate_summary_markdown
from .utils import (
    append_jsonl,
    copy_file,
    copytree,
    dump_json,
    dump_jsonl,
    ensure_dir,
    load_dotenv,
    load_json,
    load_jsonl,
    mask_secret,
    normalize_answer_text,
    now_ms,
    random_token,
    read_text,
    redact_mapping,
    require_env,
    resolve_any_env,
    rm_tree,
    safe_unlink,
    find_free_loopback_port,
    sha256_file,
    short_uid,
    sleep_seconds,
    slugify,
    utc_now_iso,
    write_text,
)


@dataclass(slots=True)
class ResolvedRuntimeEnv:
    provider_api_key: str
    provider_api_key_env_name: str
    gateway_token: str
    ov_root_api_key: str
    generator_api_base: str
    generator_model_id: str
    ov_vlm_api_base: str
    ov_vlm_model: str
    ov_embed_api_base: str
    ov_embed_model: str
    judge_api_base: str
    judge_model: str


@dataclass(slots=True)
class BlockSpec:
    rerun_index: int
    rerun_id: str
    group: GroupSpec
    sample: SampleRecord


class BenchmarkRunner:
    def __init__(self, cfg: ExperimentConfig, *, fresh: bool = False) -> None:
        self.cfg = cfg
        self.fresh = fresh
        self.run_id = f"run-{short_uid(k=12)}"
        self.repo_root = Path(__file__).resolve().parents[2]
        self.vendor_ov_conf_template = self.repo_root / "configs" / "ov.conf.template.json"
        self.vendor_judge_prompt = self.repo_root / "configs" / "judge_prompt_v1.txt"

        self.runtime_root = ensure_dir(cfg.runtime.runtime_root)
        self.artifacts_dir = ensure_dir(cfg.runtime.artifacts_dir)
        self.toolchain_dir = ensure_dir(cfg.runtime.toolchain_dir)
        self.repos_dir = ensure_dir(cfg.runtime.repos_dir)
        self.snapshots_dir = ensure_dir(cfg.runtime.snapshots_dir)
        self.work_dir = ensure_dir(cfg.runtime.work_dir)
        self.blocks_dir = ensure_dir(self.artifacts_dir / "blocks")
        self.logs_dir = ensure_dir(self.runtime_root / "logs")
        self.prepare_logs_dir = ensure_dir(self.logs_dir / "prepare")

        self.openclaw_prefix = self.toolchain_dir / "openclaw"
        self.openviking_venv = self.toolchain_dir / "openviking-venv"

        self.base_home = self.snapshots_dir / "_base"
        self.setup_repo_dir = self.repos_dir / "openclaw-openviking-doubao"
        self.dataset_repo_dir = self.repos_dir / "OpenViking-LoCoMo10"
        self.openclaw_repo_dir = self.repos_dir / "openclaw"
        self.openviking_repo_dir = self.repos_dir / "OpenViking"
        self.openclaw_eval_repo_dir = self.repos_dir / "openclaw-eval"

        self._resolved_env: ResolvedRuntimeEnv | None = None
        self._dataset: DatasetBundle | None = None
        self._manifest_cache: dict[str, Any] | None = None
        self._openclaw_bin: Path | None = None
        self._openviking_python: Path | None = None

    # ------------------------------------------------------------------
    # public entrypoints
    # ------------------------------------------------------------------

    def full_run(self) -> Path:
        self.prepare()
        self.run_all_blocks()
        self.aggregate_outputs()
        self.finalize_manifest()
        return self.artifacts_dir

    def prepare(self) -> None:
        if self.fresh:
            rm_tree(self.artifacts_dir)
            rm_tree(self.work_dir)
            ensure_dir(self.artifacts_dir)
            ensure_dir(self.work_dir)
        self._resolved_env = self.resolve_runtime_env()
        self._dataset = load_dataset(self.cfg.dataset, tail=self.cfg.runtime.ingest_tail)
        validation = validate_dataset(self._dataset, self.cfg.dataset)
        if not validation["ok"]:
            raise RuntimeError(f"Dataset validation failed: {validation}")
        self.update_manifest(
            {
                "run_id": self.run_id,
                "plan_lock": "formal-three-group-v2",
                "started_at": utc_now_iso(),
                "dataset_validation": validation,
            }
        )
        self.clone_reference_repos()
        self.install_toolchains()
        self.prepare_base_home()
        self.prepare_group_snapshots()

    def run_all_blocks(self) -> None:
        assert self._dataset is not None
        for block in self.plan_blocks():
            attempts = 0
            while True:
                attempts += 1
                try:
                    self.run_single_block(block)
                    break
                except Exception as exc:  # noqa: BLE001
                    self.write_block_status(
                        block,
                        {
                            "valid": False,
                            "attempt": attempts,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                            "finished_at": utc_now_iso(),
                        },
                    )
                    if attempts > self.cfg.runtime.max_block_retries:
                        raise

    def aggregate_outputs(self) -> None:
        sample_rows: list[dict[str, Any]] = []
        direct_rows: list[dict[str, Any]] = []
        amort_rows: list[dict[str, Any]] = []
        for status_path in sorted(self.blocks_dir.glob("**/status.json")):
            status = load_json(status_path)
            if not status.get("valid"):
                continue
            block_dir = status_path.parent
            sample_rows.extend(load_jsonl(block_dir / "sample_ingest_metrics.jsonl"))
            direct_rows.extend(load_jsonl(block_dir / "task_metrics_direct.jsonl"))
            amort_rows.extend(load_jsonl(block_dir / "task_metrics_amortized.jsonl"))

        sample_df = to_dataframe(sample_rows, SAMPLE_INGEST_COLUMNS)
        direct_df = to_dataframe(direct_rows, TASK_DIRECT_COLUMNS)
        amort_df = to_dataframe(amort_rows, TASK_AMORTIZED_COLUMNS)
        if sample_df.empty or direct_df.empty or amort_df.empty:
            raise RuntimeError("No valid block metrics were found to aggregate.")

        # strict row-count validation by group and rerun
        expected_total = self.cfg.dataset.expected_total_cases
        for (rerun_id, group_id), group_df in direct_df.groupby(["rerun_id", "group_id"]):
            if len(group_df) != expected_total:
                raise RuntimeError(
                    f"Group {group_id} / {rerun_id} has {len(group_df)} task rows, expected {expected_total}"
                )

        metrics_root = ensure_dir(self.artifacts_dir / "metrics")
        summary_root = ensure_dir(self.artifacts_dir / "summary")
        ensure_dir(metrics_root / "sample_ingest")
        ensure_dir(metrics_root / "task_direct")
        ensure_dir(metrics_root / "task_amortized")

        # write group-wise and all-groups datasets
        for group_id, group_df in sample_df.groupby("group_id"):
            group_df.to_parquet(metrics_root / "sample_ingest" / f"{group_id}.parquet", index=False)
            group_df.to_csv(metrics_root / "sample_ingest" / f"{group_id}.csv", index=False)
        for group_id, group_df in direct_df.groupby("group_id"):
            group_df.to_parquet(metrics_root / "task_direct" / f"{group_id}.parquet", index=False)
            group_df.to_csv(metrics_root / "task_direct" / f"{group_id}.csv", index=False)
        for group_id, group_df in amort_df.groupby("group_id"):
            group_df.to_parquet(metrics_root / "task_amortized" / f"{group_id}.parquet", index=False)
            group_df.to_csv(metrics_root / "task_amortized" / f"{group_id}.csv", index=False)

        direct_df.to_parquet(metrics_root / "task_metrics_direct_all_groups.parquet", index=False)
        direct_df.to_csv(metrics_root / "task_metrics_direct_all_groups.csv", index=False)
        amort_df.to_parquet(metrics_root / "task_metrics_amortized_all_groups.parquet", index=False)
        amort_df.to_csv(metrics_root / "task_metrics_amortized_all_groups.csv", index=False)

        # reconcile per group / rerun
        reconciliation: dict[str, Any] = {}
        for (rerun_id, group_id), group_sample_df in sample_df.groupby(["rerun_id", "group_id"]):
            group_amort_df = amort_df[(amort_df["rerun_id"] == rerun_id) & (amort_df["group_id"] == group_id)]
            reconciliation[f"{rerun_id}:{group_id}"] = reconcile_group_totals(group_sample_df, group_amort_df)
            if not reconciliation[f"{rerun_id}:{group_id}"]["ok"]:
                raise RuntimeError(f"Token/time reconciliation failed for {rerun_id}:{group_id}")

        summary_paths = generate_summary_markdown(
            self.cfg,
            task_direct_df=direct_df,
            task_amortized_df=amort_df,
            output_dir=summary_root,
        )
        self.materialize_canonical_raw_and_logs()
        self.update_manifest(
            {
                "aggregation": {
                    "sample_ingest_rows": int(len(sample_df)),
                    "task_direct_rows": int(len(direct_df)),
                    "task_amortized_rows": int(len(amort_df)),
                    "reconciliation": reconciliation,
                },
                "summary_files": {k: str(v.relative_to(self.artifacts_dir)) for k, v in summary_paths.items()},
            }
        )

    def finalize_manifest(self) -> None:
        self.update_manifest({"finished_at": utc_now_iso()})

    # ------------------------------------------------------------------
    # preparation
    # ------------------------------------------------------------------

    def resolve_runtime_env(self) -> ResolvedRuntimeEnv:
        provider_name, provider_key = resolve_any_env(
            self.cfg.env.shared_api_key_env,
            self.cfg.env.volcengine_api_key_env,
        )
        gateway_token = os.environ.get(self.cfg.env.gateway_token_env, "").strip() or random_token(32)
        ov_root_api_key = os.environ.get(self.cfg.env.ov_root_api_key_env, "").strip() or random_token(16)
        resolved = ResolvedRuntimeEnv(
            provider_api_key=provider_key,
            provider_api_key_env_name=provider_name,
            gateway_token=gateway_token,
            ov_root_api_key=ov_root_api_key,
            generator_api_base=require_env(self.cfg.models.generator_api_base_env),
            generator_model_id=require_env(self.cfg.models.generator_model_id_env),
            ov_vlm_api_base=require_env(self.cfg.models.ov_vlm_api_base_env),
            ov_vlm_model=require_env(self.cfg.models.ov_vlm_model_env),
            ov_embed_api_base=require_env(self.cfg.models.ov_embed_api_base_env),
            ov_embed_model=require_env(self.cfg.models.ov_embed_model_env),
            judge_api_base=os.environ.get(self.cfg.models.judge_api_base_env, "").strip()
            or require_env(self.cfg.models.generator_api_base_env),
            judge_model=require_env(self.cfg.models.judge_model_env),
        )
        self.update_manifest(
            {
                "models": {
                    "generator_alias": self.cfg.models.generator_alias,
                    "generator_runtime_id": resolved.generator_model_id,
                    "generator_api_base": resolved.generator_api_base,
                    "openviking_vlm_model": resolved.ov_vlm_model,
                    "openviking_vlm_api_base": resolved.ov_vlm_api_base,
                    "openviking_embedding_model": resolved.ov_embed_model,
                    "openviking_embedding_api_base": resolved.ov_embed_api_base,
                    "judge_model": resolved.judge_model,
                    "judge_api_base": resolved.judge_api_base,
                },
                "secrets": {
                    "provider_api_key_env_name": provider_name,
                    "provider_api_key_masked": mask_secret(provider_key),
                    "gateway_token_masked": mask_secret(gateway_token),
                    "ov_root_api_key_masked": mask_secret(ov_root_api_key),
                },
            }
        )
        return resolved

    def clone_reference_repos(self) -> None:
        clone_or_update(self.cfg.repos.setup_repo_url, self.setup_repo_dir)
        clone_or_update(self.cfg.repos.dataset_repo_url, self.dataset_repo_dir)
        clone_or_update(self.cfg.repos.openclaw_repo_url, self.openclaw_repo_dir)
        clone_or_update(self.cfg.repos.openviking_repo_url, self.openviking_repo_dir)
        clone_or_update(self.cfg.repos.openclaw_eval_repo_url, self.openclaw_eval_repo_dir)

        self.update_manifest(
            {
                "commits": {
                    "openclaw_openviking_doubao": {
                        "sha": get_commit_sha(self.setup_repo_dir),
                        "url": get_remote_url(self.setup_repo_dir),
                    },
                    "dataset_repo": {
                        "sha": get_commit_sha(self.dataset_repo_dir),
                        "url": get_remote_url(self.dataset_repo_dir),
                    },
                    "openclaw_official": {
                        "sha": get_commit_sha(self.openclaw_repo_dir),
                        "url": get_remote_url(self.openclaw_repo_dir),
                    },
                    "openviking_official": {
                        "sha": get_commit_sha(self.openviking_repo_dir),
                        "url": get_remote_url(self.openviking_repo_dir),
                    },
                    "openclaw_eval": {
                        "sha": get_commit_sha(self.openclaw_eval_repo_dir),
                        "url": get_remote_url(self.openclaw_eval_repo_dir),
                        "source_commit_from_dataset_manifest": (self._dataset.manifest.get("source") or {}).get("commit") if self._dataset else None,
                    },
                }
            }
        )

    def install_toolchains(self) -> None:
        assert self._resolved_env is not None
        self._openclaw_bin = install_openclaw(
            self.openclaw_prefix,
            self.cfg.versions.openclaw,
            self.prepare_logs_dir / "install_openclaw.log",
        )
        self._openviking_python = install_openviking(
            self.openviking_venv,
            self.cfg.versions.openviking,
            self.prepare_logs_dir / "install_openviking.log",
        )
        host_info = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "node": self._safe_cmd_output([str(self._openclaw_bin), "--version"]),
            "openclaw_version_runtime": get_openclaw_version(self._openclaw_bin),
            "openviking_version_runtime": get_openviking_version(self._openviking_python),
        }
        if host_info["openclaw_version_runtime"] != self.cfg.versions.openclaw:
            raise RuntimeError(
                f"OpenClaw runtime version mismatch: expected {self.cfg.versions.openclaw}, got {host_info['openclaw_version_runtime']}"
            )
        if host_info["openviking_version_runtime"] != self.cfg.versions.openviking:
            raise RuntimeError(
                f"OpenViking runtime version mismatch: expected {self.cfg.versions.openviking}, got {host_info['openviking_version_runtime']}"
            )
        self.update_manifest({"host": host_info, "versions": {"openclaw": self.cfg.versions.openclaw, "openviking": self.cfg.versions.openviking}})

    def prepare_base_home(self) -> None:
        assert self._openclaw_bin is not None
        assert self._resolved_env is not None
        if self.base_home.exists() and not self.fresh:
            return
        rm_tree(self.base_home)
        ensure_dir(self.base_home / ".openclaw")
        ensure_dir(self.base_home / ".openviking")
        env = self.build_process_env(self.base_home)
        self.write_runtime_env_files(self.base_home)
        copy_file(self.vendor_ov_conf_template, self.base_home / ".openviking" / "ov.conf.template.json")
        onboard_custom_provider(
            self._openclaw_bin,
            env=env,
            base_url=self._resolved_env.generator_api_base,
            model_id=self._resolved_env.generator_model_id,
            custom_compatibility=self.cfg.env.custom_api_compatibility,
        )
        install_plugin(self._openclaw_bin, self.setup_repo_dir / "plugin", env=env)
        write_text(self.prepare_logs_dir / "plugin_inspect.txt", inspect_plugin(self._openclaw_bin, env=env))
        save_redacted_openclaw_config(self.base_home / ".openclaw" / "openclaw.json", self.prepare_logs_dir / "base_openclaw.redacted.json")

    def prepare_group_snapshots(self) -> None:
        assert self._openclaw_bin is not None
        for group in self.cfg.groups:
            snapshot_dir = self.group_snapshot_dir(group)
            if snapshot_dir.exists() and not self.fresh:
                if (snapshot_dir / ".openclaw" / "openclaw.json").is_file():
                    continue
                rm_tree(snapshot_dir)
            rm_tree(snapshot_dir)
            copytree(self.base_home, snapshot_dir)
            env = self.build_process_env(snapshot_dir)
            self.write_runtime_env_files(snapshot_dir)
            apply_group_config(
                self._openclaw_bin,
                env=env,
                runtime=self.cfg.runtime,
                group=group,
                ov_conf_path=snapshot_dir / ".openviking" / "ov.conf",
            )
            group_config_dir = ensure_dir(self.artifacts_dir / "configs")
            save_redacted_openclaw_config(
                snapshot_dir / ".openclaw" / "openclaw.json",
                group_config_dir / f"group-{group.id.lower()}-{group.slug}.openclaw.json",
            )
            render_ov_conf(
                snapshot_dir / ".openviking" / "ov.conf.template.json",
                group_config_dir / f"group-{group.id.lower()}-{group.slug}.ov.conf",
                self.ov_template_values(snapshot_dir),
                redact_api_keys=True,
            )
            if group.is_ov and self.cfg.runtime.run_ov_smoke_test:
                self.run_ov_smoke_test(snapshot_dir, group)

    # ------------------------------------------------------------------
    # block execution
    # ------------------------------------------------------------------

    def plan_blocks(self) -> list[BlockSpec]:
        assert self._dataset is not None
        groups = self.cfg.groups
        blocks: list[BlockSpec] = []
        for rerun_index in range(self.cfg.runtime.reruns):
            rerun_id = f"R{rerun_index + 1}"
            for sample_idx, sample in enumerate(self._dataset.samples):
                shift = (sample_idx + rerun_index) % len(groups)
                ordered_groups = groups[shift:] + groups[:shift]
                for group in ordered_groups:
                    blocks.append(BlockSpec(rerun_index=rerun_index, rerun_id=rerun_id, group=group, sample=sample))
        return blocks

    def run_single_block(self, block: BlockSpec) -> None:
        status_path = self.block_dir(block) / "status.json"
        if self.cfg.runtime.resume and status_path.exists():
            status = load_json(status_path)
            if status.get("valid"):
                return

        group = block.group
        sample = block.sample
        snapshot_dir = self.group_snapshot_dir(group)
        workdir = self.block_workdir(block)
        rm_tree(workdir)
        copytree(snapshot_dir, workdir)
        ensure_dir(self.block_dir(block))
        self.write_runtime_env_files(workdir)
        render_ov_conf(
            workdir / ".openviking" / "ov.conf.template.json",
            workdir / ".openviking" / "ov.conf",
            self.ov_template_values(workdir),
            redact_api_keys=False,
        )
        env = self.build_process_env(workdir)
        apply_group_config(
            self._openclaw_bin,
            env=env,
            runtime=self.cfg.runtime,
            group=group,
            ov_conf_path=workdir / ".openviking" / "ov.conf",
        )
        gateway_listen_port = find_free_loopback_port()
        config_set(self._openclaw_bin, "gateway.port", gateway_listen_port, env=env)
        config_validate(self._openclaw_bin, env=env)

        gateway_log_path = workdir / "gateway.stdout.log"
        ov_log_path = self.openviking_log_path(workdir, ensure_parent=True)

        gateway_proc = start_gateway(self._openclaw_bin, env=env, log_path=gateway_log_path)
        gateway_base_url = None
        raw_ingest_rows: list[dict[str, Any]] = []
        raw_qa_rows: list[dict[str, Any]] = []
        direct_rows: list[dict[str, Any]] = []
        ingest_gateway_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        ingest_session_ids: list[str] = []
        barrier_result_map: dict[str, Any] = {}
        ov_inspector: OpenVikingInspector | None = None
        block_dir = self.block_dir(block)

        try:
            gateway_base_url = wait_for_gateway_ready(
                self._openclaw_bin,
                env=env,
                timeout_seconds=self.cfg.runtime.gateway_start_timeout_seconds,
                poll_seconds=self.cfg.runtime.gateway_health_poll_seconds,
                fallback_base_url=f"http://127.0.0.1:{gateway_listen_port}",
            )
            assert_group_runtime(self._openclaw_bin, env=env, group=group)
            if group.is_ov:
                ov_inspector = OpenVikingInspector(
                    f"http://127.0.0.1:{self.cfg.runtime.openviking_port}",
                    api_key=self._resolved_env.ov_root_api_key,
                    agent_id=self.ov_http_agent_id(),
                )
                # wait for local mode server to be reachable
                self.wait_for_ov_health(ov_inspector)

            ingest_start_iso = utc_now_iso()
            ingest_start_ms = now_ms()

            # Ingest each session in order with reset after each.
            for sess in sample.sessions:
                req_start_iso = utc_now_iso()
                req_start_ms = now_ms()
                body = post_response(
                    gateway_base_url,
                    self._resolved_env.gateway_token,
                    user=self.deterministic_user_key(block),
                    message=sess.message,
                    timeout_seconds=self.cfg.runtime.request_timeout_seconds,
                    max_retries=self.cfg.runtime.qa_retry_count,
                    retry_backoff_seconds=self.cfg.runtime.qa_retry_backoff_seconds,
                )
                req_end_iso = utc_now_iso()
                req_end_ms = now_ms()
                usage = body.get("usage") or {}
                for key in ("input_tokens", "output_tokens", "total_tokens"):
                    ingest_gateway_usage[key] += int(usage.get(key, 0))
                session_id = self.wait_for_session_id(workdir, self.deterministic_user_key(block))
                if not session_id:
                    raise RuntimeError(f"Could not resolve OpenClaw session id during ingest for {block.group.id}/{sample.sample_id}")
                ingest_session_ids.append(session_id)
                archived = archive_session_file(workdir / ".openclaw", self.cfg.runtime.openclaw_agent_id, session_id)
                raw_ingest_rows.append(
                    {
                        "run_id": self.run_id,
                        "group_id": group.id,
                        "rerun_id": block.rerun_id,
                        "sample_id": sample.sample_id,
                        "session_key": sess.session_key,
                        "date_time": sess.date_time,
                        "request_start_ts": req_start_iso,
                        "request_end_ts": req_end_iso,
                        "request_elapsed_ms": req_end_ms - req_start_ms,
                        "session_id": session_id,
                        "archived_session_file": archived,
                        "usage": usage,
                        "response_text": body,
                    }
                )

            ov_barrier_wait_ms = 0
            post_reset_quiet_wait_ms = 0
            explicit_commit_usages: list[OvUsage] = []

            if group.is_ov:
                unique_session_ids = []
                for sid in ingest_session_ids:
                    if sid not in unique_session_ids:
                        unique_session_ids.append(sid)
                barrier_started_ms = now_ms()
                barrier_deadline = barrier_started_ms / 1000.0 + self.cfg.runtime.ov_barrier_timeout_seconds
                for sid in unique_session_ids:
                    remaining = max(1.0, barrier_deadline - time.time())
                    result = wait_for_commit_visibility(
                        ov_inspector,
                        sid,
                        timeout_seconds=remaining,
                        poll_seconds=5.0,
                        explicit_commit_fallback=self.cfg.runtime.fallback_explicit_commit_for_telemetry,
                        require_extracted_memories=self.cfg.runtime.barrier_require_extracted_memories,
                    )
                    barrier_result_map[sid] = result
                    explicit_commit_usages.append(usage_from_payload(result.explicit_commit_payload, source="explicit_commit"))
                    commit_ok = isinstance(result.detail, dict) and int(result.detail.get("commit_count", 0) or 0) > 0
                    overview_ok = isinstance(result.context, dict) and bool(str(result.context.get("latest_archive_overview") or "").strip())
                    memory_ok = extract_memory_total(result.detail) > 0
                    if not self.cfg.runtime.barrier_require_extracted_memories:
                        memory_ok = True
                    if not (commit_ok and overview_ok and memory_ok):
                        raise RuntimeError(
                            f"OV barrier not satisfied within {self.cfg.runtime.ov_barrier_timeout_seconds}s for {group.id}/{sample.sample_id}/{sid}"
                        )
                ov_barrier_wait_ms = now_ms() - barrier_started_ms
            else:
                sleep_seconds(self.cfg.runtime.noov_quiet_wait_seconds)
                post_reset_quiet_wait_ms = int(self.cfg.runtime.noov_quiet_wait_seconds * 1000)

            ingest_end_iso = utc_now_iso()
            ingest_end_ms = now_ms()

            # QA phase.
            qa_windows: list[dict[str, Any]] = []
            for case in sample.cases:
                retries_used = 0
                qa_error_flag = False
                body: dict[str, Any] | None = None
                prediction = ""
                last_error = None
                qa_start_iso = utc_now_iso()
                qa_start_ms = now_ms()
                while True:
                    try:
                        body = post_response(
                            gateway_base_url,
                            self._resolved_env.gateway_token,
                            user=self.deterministic_user_key(block),
                            message=case.question,
                            timeout_seconds=self.cfg.runtime.request_timeout_seconds,
                            max_retries=0,
                        )
                        prediction = self.extract_prediction(body)
                        break
                    except Exception as exc:  # noqa: BLE001
                        last_error = exc
                        if retries_used >= self.cfg.runtime.qa_retry_count:
                            qa_error_flag = True
                            prediction = f"[ERROR] {exc}"
                            body = {"usage": {}, "error": str(exc)}
                            break
                        retries_used += 1
                        sleep_seconds(self.cfg.runtime.qa_retry_backoff_seconds * retries_used)
                qa_end_iso = utc_now_iso()
                qa_end_ms = now_ms()
                session_id = self.wait_for_session_id(workdir, self.deterministic_user_key(block))
                if session_id:
                    archive_session_file(workdir / ".openclaw", self.cfg.runtime.openclaw_agent_id, session_id)
                raw_qa_rows.append(
                    {
                        "run_id": self.run_id,
                        "group_id": group.id,
                        "rerun_id": block.rerun_id,
                        "sample_id": sample.sample_id,
                        "case_uid": case.case_uid,
                        "question": case.question,
                        "gold_answer": case.gold_answer,
                        "request_start_ts": qa_start_iso,
                        "request_end_ts": qa_end_iso,
                        "request_elapsed_ms": qa_end_ms - qa_start_ms,
                        "retry_count": retries_used,
                        "error_flag": qa_error_flag,
                        "session_id": session_id,
                        "response_body": body,
                    }
                )
                qa_windows.append(
                    {
                        "case_uid": case.case_uid,
                        "case_id": case.case_id,
                        "sample_idx": case.sample_idx,
                        "qa_idx_within_sample": case.qa_idx_within_sample,
                        "category": case.category,
                        "question": case.question,
                        "gold_answer": case.gold_answer,
                        "prediction": prediction,
                        "qa_start_ts": qa_start_iso,
                        "qa_end_ts": qa_end_iso,
                        "qa_start_ms": qa_start_ms,
                        "qa_end_ms": qa_end_ms,
                        "qa_elapsed_ms": qa_end_ms - qa_start_ms,
                        "qa_retry_count": retries_used,
                        "qa_error_flag": qa_error_flag,
                        "qa_session_id": session_id,
                        "gateway_usage": (body or {}).get("usage") or {},
                    }
                )

        finally:
            gateway_proc.terminate_tree()
            # copy live logs out before optional workdir cleanup
            self.capture_block_logs_and_configs(block, workdir)

        ov_events = parse_ov_log(ov_log_path) if group.is_ov else []
        ingest_log_usage = aggregate_usage_from_events(
            ov_events,
            start_ms=ingest_start_ms,
            end_ms=ingest_end_ms,
            session_id=None,
            slop_seconds=self.cfg.runtime.telemetry_slop_seconds,
        ) if group.is_ov else OvUsage(source="not_applicable")
        ingest_commit_usage = merge_usage(*explicit_commit_usages) if explicit_commit_usages else OvUsage(source="none")
        ingest_ov_usage = merge_usage(ingest_log_usage, ingest_commit_usage) if group.is_ov else OvUsage(source="not_applicable")

        if group.is_ov and self.cfg.runtime.strict_ov_usage and ingest_ov_usage.total_tokens <= 0:
            raise RuntimeError(f"Strict OV usage mode: ingest OV internal usage missing for {group.id}/{sample.sample_id}")

        # Judge after SUT is done
        judge_cfg = JudgeConfig(
            base_url=self._resolved_env.judge_api_base,
            api_key=self._resolved_env.provider_api_key,
            model=self._resolved_env.judge_model,
            prompt_version="judge_prompt_v1",
            prompt_text=read_text(self.vendor_judge_prompt),
            parallel=self.cfg.runtime.judge_parallel,
        )
        judge_input = [{"case_uid": row["case_uid"], "question": row["question"], "gold_answer": row["gold_answer"], "prediction": row["prediction"]} for row in qa_windows]
        judge_rows = judge_records(judge_input, judge_cfg)
        judge_by_case = {row["case_uid"]: row for row in judge_rows}

        # build metrics rows
        last_sid = ingest_session_ids[-1] if ingest_session_ids else None
        last_barrier = barrier_result_map.get(last_sid) if last_sid else None
        sample_ingest_record = build_sample_ingest_record(
            run_id=self.run_id,
            group_id=group.id,
            rerun_id=block.rerun_id,
            sample_id=sample.sample_id,
            sessions_ingested=len(sample.sessions),
            ingest_start_ts=ingest_start_iso,
            ingest_end_ts=ingest_end_iso,
            ingest_elapsed_ms=ingest_end_ms - ingest_start_ms,
            gateway_usage=ingest_gateway_usage,
            ov_usage=ingest_ov_usage,
            ov_barrier_wait_ms=ov_barrier_wait_ms,
            post_reset_quiet_wait_ms=post_reset_quiet_wait_ms,
            barrier_session_id=last_sid,
            barrier_detail=(last_barrier.detail if last_barrier else None),
            barrier_context=(last_barrier.context if last_barrier else None),
        )

        for row in qa_windows:
            ov_usage = aggregate_usage_from_events(
                ov_events,
                start_ms=row["qa_start_ms"],
                end_ms=row["qa_end_ms"],
                session_id=row["qa_session_id"] if group.is_ov else None,
                slop_seconds=self.cfg.runtime.telemetry_slop_seconds,
            ) if group.is_ov else OvUsage(source="not_applicable")
            if group.is_ov and self.cfg.runtime.strict_ov_usage and ov_usage.total_tokens <= 0:
                raise RuntimeError(
                    f"Strict OV usage mode: per-QA OV internal usage missing for {group.id}/{sample.sample_id}/{row['case_uid']}"
                )
            judge = judge_by_case[row["case_uid"]]
            direct_rows.append(
                build_task_direct_record(
                    run_id=self.run_id,
                    group_id=group.id,
                    rerun_id=block.rerun_id,
                    sample_id=sample.sample_id,
                    case_uid=row["case_uid"],
                    case_id=row["case_id"],
                    sample_idx=row["sample_idx"],
                    qa_idx_within_sample=row["qa_idx_within_sample"],
                    category=row["category"],
                    question=row["question"],
                    gold_answer=row["gold_answer"],
                    prediction=row["prediction"],
                    judge_correct=judge["is_correct"],
                    judge_reasoning_raw=judge["reasoning"],
                    judge_model_id=judge["judge_model_id"],
                    judge_prompt_version=judge["judge_prompt_version"],
                    qa_start_ts=row["qa_start_ts"],
                    qa_end_ts=row["qa_end_ts"],
                    qa_elapsed_ms=row["qa_elapsed_ms"],
                    qa_retry_count=row["qa_retry_count"],
                    qa_error_flag=row["qa_error_flag"],
                    gateway_usage=row["gateway_usage"],
                    ov_usage=ov_usage,
                    qa_session_id=row["qa_session_id"],
                )
            )

        amort_rows = amortize_tasks(direct_rows, sample_ingest_record)
        self.write_block_outputs(
            block,
            sample_ingest_record=sample_ingest_record,
            direct_rows=direct_rows,
            amort_rows=amort_rows,
            raw_ingest_rows=raw_ingest_rows,
            raw_qa_rows=raw_qa_rows,
            judge_rows=judge_rows,
        )
        self.write_block_status(
            block,
            {
                "valid": True,
                "attempt": 1,
                "sample_id": sample.sample_id,
                "group_id": group.id,
                "rerun_id": block.rerun_id,
                "finished_at": utc_now_iso(),
            },
        )

        if not self.cfg.runtime.keep_workdirs:
            rm_tree(workdir)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def deterministic_user_key(self, block: BlockSpec) -> str:
        return f"{block.group.id.lower()}-{block.group.slug}-{block.rerun_id.lower()}-{block.sample.sample_id}"

    def ov_http_agent_id(self) -> str:
        """X-OpenViking-Agent for direct OpenViking HTTP calls.

        When plugin config ``agentId`` is the literal ``default``, the openviking plugin
        sends OpenClaw ``ctx.agentId`` (e.g. ``main``), not the string ``default``.
        """
        if self.cfg.runtime.openviking_agent_id == "default":
            return self.cfg.runtime.openclaw_agent_id
        return self.cfg.runtime.openviking_agent_id

    def group_snapshot_dir(self, group: GroupSpec) -> Path:
        return self.snapshots_dir / f"{group.id.lower()}-{group.slug}"

    def block_dir(self, block: BlockSpec) -> Path:
        return self.blocks_dir / block.rerun_id / block.group.id / block.sample.sample_id

    def block_workdir(self, block: BlockSpec) -> Path:
        return self.work_dir / block.rerun_id / block.group.id / block.sample.sample_id

    def openviking_log_path(self, workdir: Path, *, ensure_parent: bool = False) -> Path:
        """Resolve OpenViking runtime log path across layout variants."""
        primary = workdir / ".openviking" / "log" / "openviking.log"
        legacy = workdir / ".openviking" / "data" / "log" / "openviking.log"
        if primary.exists():
            return primary
        if legacy.exists():
            return legacy
        if ensure_parent:
            ensure_dir(primary.parent)
        return primary

    def build_process_env(self, home_root: str | Path) -> dict[str, str]:
        assert self._resolved_env is not None
        env = os.environ.copy()
        home_root = Path(home_root)
        env["HOME"] = str(home_root)
        env["OPENCLAW_HOME"] = str(home_root / ".openclaw")
        env["OPENCLAW_STATE_DIR"] = str(home_root / ".openclaw")
        env["OPENCLAW_CONFIG_PATH"] = str(home_root / ".openclaw" / "openclaw.json")
        env["OPENCLAW_GATEWAY_TOKEN"] = self._resolved_env.gateway_token
        env[self.cfg.env.custom_api_key_env] = self._resolved_env.provider_api_key
        env[self.cfg.env.volcengine_api_key_env] = self._resolved_env.provider_api_key
        env[self.cfg.env.ov_root_api_key_env] = self._resolved_env.ov_root_api_key
        env["OPENVIKING_CONFIG_FILE"] = str(home_root / ".openviking" / "ov.conf")
        env["OPENVIKING_PYTHON"] = str(self._openviking_python)
        env["OPENVIKING_API_KEY"] = self._resolved_env.ov_root_api_key
        env["PATH"] = f"{self.openclaw_prefix / 'bin'}:{env.get('PATH', '')}"
        return env

    def write_runtime_env_files(self, home_root: str | Path) -> None:
        assert self._resolved_env is not None
        home_root = Path(home_root)
        write_openclaw_dotenv(
            home_root,
            {
                self.cfg.env.custom_api_key_env: self._resolved_env.provider_api_key,
                self.cfg.env.volcengine_api_key_env: self._resolved_env.provider_api_key,
                "OPENCLAW_GATEWAY_TOKEN": self._resolved_env.gateway_token,
                "OPENVIKING_API_KEY": self._resolved_env.ov_root_api_key,
            },
        )
        write_text(
            home_root / ".openclaw" / "openviking.env",
            "\n".join(
                [
                    f"export OPENVIKING_PYTHON={self._openviking_python}",
                    f"export OPENVIKING_CONFIG_FILE={home_root / '.openviking' / 'ov.conf'}",
                    f"export OPENVIKING_API_KEY={self._resolved_env.ov_root_api_key}",
                ]
            )
            + "\n",
        )

    def ov_template_values(self, home_root: str | Path) -> dict[str, Any]:
        assert self._resolved_env is not None
        home_root = Path(home_root)
        return {
            "OPENVIKING_HOME": str(home_root / ".openviking"),
            "OPENVIKING_PORT": self.cfg.runtime.openviking_port,
            "OPENVIKING_ROOT_API_KEY": self._resolved_env.ov_root_api_key,
            "OV_PROVIDER_API_KEY": self._resolved_env.provider_api_key,
            "OV_VLM_API_BASE": self._resolved_env.ov_vlm_api_base,
            "OV_VLM_MODEL": self._resolved_env.ov_vlm_model,
            "OV_EMBED_API_BASE": self._resolved_env.ov_embed_api_base,
            "OV_EMBED_MODEL": self._resolved_env.ov_embed_model,
        }

    def wait_for_ov_health(self, inspector: OpenVikingInspector, timeout_seconds: float = 90.0) -> None:
        import time as _time

        deadline = _time.time() + timeout_seconds
        last_error = None
        while _time.time() < deadline:
            try:
                payload = inspector.health()
                if isinstance(payload, dict) and (payload.get("healthy") is True or payload.get("status") == "ok"):
                    return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
            sleep_seconds(2.0)
        raise RuntimeError(f"OpenViking local mode service did not become healthy: {last_error}")

    def wait_for_session_id(self, home_root: str | Path, user_key: str, timeout_seconds: float = 5.0) -> str | None:
        import time as _time

        deadline = _time.time() + timeout_seconds
        while _time.time() < deadline:
            sid = get_session_id_for_user(Path(home_root) / ".openclaw", self.cfg.runtime.openclaw_agent_id, user_key)
            if sid:
                return sid
            sleep_seconds(0.25)
        return None

    def extract_prediction(self, body: dict[str, Any]) -> str:
        from .openclaw import extract_response_text

        text = extract_response_text(body)
        return text if text else json.dumps(body, ensure_ascii=False)

    def run_ov_smoke_test(self, snapshot_dir: Path, group: GroupSpec) -> None:
        smoke_workdir = self.work_dir / "smoke" / group.id
        rm_tree(smoke_workdir)
        copytree(snapshot_dir, smoke_workdir)
        self.write_runtime_env_files(smoke_workdir)
        render_ov_conf(
            smoke_workdir / ".openviking" / "ov.conf.template.json",
            smoke_workdir / ".openviking" / "ov.conf",
            self.ov_template_values(smoke_workdir),
            redact_api_keys=False,
        )
        env = self.build_process_env(smoke_workdir)
        apply_group_config(
            self._openclaw_bin,
            env=env,
            runtime=self.cfg.runtime,
            group=group,
            ov_conf_path=smoke_workdir / ".openviking" / "ov.conf",
        )
        gateway_listen_port = find_free_loopback_port()
        config_set(self._openclaw_bin, "gateway.port", gateway_listen_port, env=env)
        if group.is_ov:
            # Default plugin threshold is 20k pending tokens before auto-commit; short smoke turns
            # stay far below that, so auto-commit never runs. Lower only for this ephemeral workdir.
            config_set(
                self._openclaw_bin,
                "plugins.entries.openviking.config.commitTokenThreshold",
                128,
                env=env,
            )
        config_validate(self._openclaw_bin, env=env)
        proc = start_gateway(self._openclaw_bin, env=env, log_path=smoke_workdir / "gateway.smoke.log")
        probe = f"probe-{short_uid(k=8)}"
        user = f"smoke-{group.id.lower()}-{probe}"
        try:
            base_url = wait_for_gateway_ready(
                self._openclaw_bin,
                env=env,
                timeout_seconds=self.cfg.runtime.gateway_start_timeout_seconds,
                poll_seconds=self.cfg.runtime.gateway_health_poll_seconds,
                fallback_base_url=f"http://127.0.0.1:{gateway_listen_port}",
            )
            assert_group_runtime(self._openclaw_bin, env=env, group=group)
            inspector = OpenVikingInspector(
                f"http://127.0.0.1:{self.cfg.runtime.openviking_port}",
                api_key=self._resolved_env.ov_root_api_key,
                agent_id=self.ov_http_agent_id(),
            )
            self.wait_for_ov_health(inspector)
            for msg in [
                f"[probe={probe}] Please remember that my favorite storage engine is PostgreSQL.",
                f"What storage engine did I ask you to remember? probe={probe}",
            ]:
                post_response(
                    base_url,
                    self._resolved_env.gateway_token,
                    user=user,
                    message=msg,
                    timeout_seconds=self.cfg.runtime.request_timeout_seconds,
                    max_retries=self.cfg.runtime.qa_retry_count,
                    retry_backoff_seconds=self.cfg.runtime.qa_retry_backoff_seconds,
                )
                self.wait_for_session_id(smoke_workdir, user)
            sid = self.wait_for_session_id(smoke_workdir, user)
            sid = sid or get_session_id_for_user(smoke_workdir / ".openclaw", self.cfg.runtime.openclaw_agent_id, user)
            if not sid:
                raise RuntimeError("Smoke test could not resolve session id.")
            barrier = wait_for_commit_visibility(
                inspector,
                sid,
                timeout_seconds=float(self.cfg.runtime.ov_barrier_timeout_seconds),
                poll_seconds=2.0,
                explicit_commit_fallback=True,
                require_extracted_memories=False,
            )
            search_deadline = time.time() + 90.0
            memories: list[Any] | None = None
            while time.time() < search_deadline:
                for query in (probe, "PostgreSQL"):
                    search_payload = inspector.search_memories(query, telemetry=True) or {}
                    cand = search_payload.get("memories") if isinstance(search_payload, dict) else None
                    if isinstance(cand, list) and cand:
                        memories = cand
                        break
                if memories:
                    break
                sleep_seconds(3.0)
            if not memories:
                write_text(
                    self.prepare_logs_dir / f"smoke-{group.id}-search-warning.txt",
                    "search_memories returned empty after 90s; commit+archive barrier still passed.\n",
                )
        finally:
            try:
                gw_log = smoke_workdir / "gateway.smoke.log"
                if gw_log.exists():
                    copy_file(gw_log, self.prepare_logs_dir / f"smoke-{group.id}-gateway.log")
                ov_log = self.openviking_log_path(smoke_workdir)
                if ov_log.exists():
                    copy_file(ov_log, self.prepare_logs_dir / f"smoke-{group.id}-openviking.log")
            except OSError:
                pass
            proc.terminate_tree()
            rm_tree(smoke_workdir)

    def capture_block_logs_and_configs(self, block: BlockSpec, workdir: Path) -> None:
        block_dir = self.block_dir(block)
        ensure_dir(block_dir)
        for src, dst_name in [
            (workdir / "gateway.stdout.log", "openclaw.log"),
            (self.openviking_log_path(workdir), "openviking.log"),
        ]:
            if src.exists():
                copy_file(src, block_dir / dst_name)
        if (workdir / ".openclaw" / "openclaw.json").exists():
            save_redacted_openclaw_config(workdir / ".openclaw" / "openclaw.json", block_dir / "openclaw.redacted.json")
        if (workdir / ".openviking" / "ov.conf.template.json").exists():
            render_ov_conf(
                workdir / ".openviking" / "ov.conf.template.json",
                block_dir / "ov.redacted.conf",
                self.ov_template_values(workdir),
                redact_api_keys=True,
            )

    def write_block_outputs(
        self,
        block: BlockSpec,
        *,
        sample_ingest_record: dict[str, Any],
        direct_rows: list[dict[str, Any]],
        amort_rows: list[dict[str, Any]],
        raw_ingest_rows: list[dict[str, Any]],
        raw_qa_rows: list[dict[str, Any]],
        judge_rows: list[dict[str, Any]],
    ) -> None:
        block_dir = self.block_dir(block)
        ensure_dir(block_dir)
        dump_jsonl([sample_ingest_record], block_dir / "sample_ingest_metrics.jsonl")
        dump_jsonl(direct_rows, block_dir / "task_metrics_direct.jsonl")
        dump_jsonl(amort_rows, block_dir / "task_metrics_amortized.jsonl")
        dump_json(raw_ingest_rows, block_dir / "ingest_raw.json")
        dump_jsonl(raw_qa_rows, block_dir / "qa_raw.jsonl")
        dump_jsonl(judge_rows, block_dir / "judge_raw.jsonl")

    def write_block_status(self, block: BlockSpec, payload: dict[str, Any]) -> None:
        block_dir = self.block_dir(block)
        ensure_dir(block_dir)
        dump_json(payload, block_dir / "status.json")
        self.update_manifest({"blocks": {f"{block.rerun_id}:{block.group.id}:{block.sample.sample_id}": payload}})

    def materialize_canonical_raw_and_logs(self) -> None:
        raw_root = ensure_dir(self.artifacts_dir / "raw")
        logs_root = ensure_dir(self.artifacts_dir / "logs")
        for status_path in sorted(self.blocks_dir.glob("**/status.json")):
            status = load_json(status_path)
            if not status.get("valid"):
                continue
            block_dir = status_path.parent
            parts = block_dir.relative_to(self.blocks_dir).parts
            rerun_id, group_id, sample_id = parts[0], parts[1], parts[2]
            ingest_src = block_dir / "ingest_raw.json"
            qa_src = block_dir / "qa_raw.jsonl"
            judge_src = block_dir / "judge_raw.jsonl"
            oc_log = block_dir / "openclaw.log"
            ov_log = block_dir / "openviking.log"
            if ingest_src.exists():
                copy_file(ingest_src, raw_root / "ingest" / group_id / rerun_id / f"{sample_id}.json")
            if qa_src.exists():
                copy_file(qa_src, raw_root / "qa" / group_id / rerun_id / f"{sample_id}.jsonl")
            if judge_src.exists():
                copy_file(judge_src, raw_root / "judge_raw" / group_id / rerun_id / f"{sample_id}.jsonl")
            if oc_log.exists():
                copy_file(oc_log, logs_root / "openclaw" / group_id / rerun_id / f"{sample_id}.log")
            if ov_log.exists():
                copy_file(ov_log, logs_root / "openviking" / group_id / rerun_id / f"{sample_id}.log")

    def update_manifest(self, patch: dict[str, Any]) -> None:
        manifest_path = self.artifacts_dir / "manifest.json"
        if self._manifest_cache is None:
            if manifest_path.exists():
                self._manifest_cache = load_json(manifest_path)
            else:
                self._manifest_cache = {}
        self._deep_merge(self._manifest_cache, patch)
        dump_json(self._manifest_cache, manifest_path)

    def _deep_merge(self, dst: dict[str, Any], src: dict[str, Any]) -> None:
        for key, value in src.items():
            if isinstance(value, dict) and isinstance(dst.get(key), dict):
                self._deep_merge(dst[key], value)
            else:
                dst[key] = value

    def _safe_cmd_output(self, args: list[str]) -> str:
        result = run_cmd(args, check=False)
        return (result.stdout or result.stderr).strip()


def load_runner(config_path: str | Path, *, fresh: bool = False) -> BenchmarkRunner:
    cfg = load_experiment_config(config_path)
    return BenchmarkRunner(cfg, fresh=fresh)
