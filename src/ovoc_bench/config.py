from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import os
import re
import yaml


@dataclass(slots=True)
class RepoConfig:
    dataset_repo_url: str
    setup_repo_url: str
    openviking_repo_url: str
    openclaw_repo_url: str
    openclaw_eval_repo_url: str


@dataclass(slots=True)
class RuntimeConfig:
    runtime_root: Path
    artifacts_dir: Path
    toolchain_dir: Path
    repos_dir: Path
    snapshots_dir: Path
    work_dir: Path
    keep_workdirs: bool = False
    resume: bool = True
    reruns: int = 1
    request_timeout_seconds: int = 300
    ov_barrier_timeout_seconds: int = 300
    noov_quiet_wait_seconds: int = 3
    gateway_start_timeout_seconds: int = 120
    gateway_health_poll_seconds: float = 2.0
    qa_retry_count: int = 2
    qa_retry_backoff_seconds: float = 2.0
    judge_parallel: int = 10
    strict_ov_usage: bool = True
    fallback_explicit_commit_for_telemetry: bool = False
    barrier_require_extracted_memories: bool = False
    run_ov_smoke_test: bool = True
    max_block_retries: int = 2
    openclaw_agent_id: str = "main"
    openviking_agent_id: str = "default"
    openviking_port: int = 1933
    gateway_base_url_fallback: str = "http://127.0.0.1:18789"
    telemetry_slop_seconds: float = 1.0
    ingest_tail: str = "[remember what's said, keep existing memory]"


@dataclass(slots=True)
class VersionConfig:
    openclaw: str
    openviking: str


@dataclass(slots=True)
class ModelEnvConfig:
    generator_alias: str
    generator_api_base_env: str
    generator_model_id_env: str
    ov_vlm_api_base_env: str
    ov_vlm_model_env: str
    ov_embed_api_base_env: str
    ov_embed_model_env: str
    judge_api_base_env: str
    judge_model_env: str


@dataclass(slots=True)
class EnvConfig:
    shared_api_key_env: str
    volcengine_api_key_env: str
    custom_api_key_env: str
    gateway_token_env: str
    ov_root_api_key_env: str
    custom_api_compatibility: str = "openai"


@dataclass(slots=True)
class GroupSpec:
    id: str
    label: str
    memory_slot: str
    context_engine: str
    openviking_enabled: bool
    deny: list[str] = field(default_factory=list)

    @property
    def slug(self) -> str:
        value = self.label.lower()
        value = re.sub(r"[^a-z0-9]+", "-", value)
        value = re.sub(r"-+", "-", value).strip("-")
        return value

    @property
    def is_ov(self) -> bool:
        return self.openviking_enabled and self.context_engine == "openviking"


@dataclass(slots=True)
class DatasetConfig:
    vendor_json: Path
    vendor_jsonl: Path
    vendor_manifest: Path
    expected_total_cases: int = 1540
    expected_sample_count: int = 10
    expected_removed_category: int = 5


@dataclass(slots=True)
class ExperimentConfig:
    name: str
    repos: RepoConfig
    runtime: RuntimeConfig
    versions: VersionConfig
    models: ModelEnvConfig
    env: EnvConfig
    dataset: DatasetConfig
    groups: list[GroupSpec]

    def group_map(self) -> dict[str, GroupSpec]:
        return {g.id: g for g in self.groups}


def _as_path(root: Path, value: str) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = (root / p).resolve()
    return p


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} did not decode to a mapping.")
    return data


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    cfg_path = Path(path).resolve()
    raw = _load_yaml(cfg_path)
    root = cfg_path.parent

    runtime_raw = raw["runtime"]
    runtime = RuntimeConfig(
        runtime_root=_as_path(root, runtime_raw["runtime_root"]),
        artifacts_dir=_as_path(root, runtime_raw["artifacts_dir"]),
        toolchain_dir=_as_path(root, runtime_raw["toolchain_dir"]),
        repos_dir=_as_path(root, runtime_raw["repos_dir"]),
        snapshots_dir=_as_path(root, runtime_raw["snapshots_dir"]),
        work_dir=_as_path(root, runtime_raw["work_dir"]),
        keep_workdirs=bool(runtime_raw.get("keep_workdirs", False)),
        resume=bool(runtime_raw.get("resume", True)),
        reruns=int(os.environ.get("OV_OC_RERUNS", runtime_raw.get("reruns", 1))),
        request_timeout_seconds=int(runtime_raw.get("request_timeout_seconds", 300)),
        ov_barrier_timeout_seconds=int(runtime_raw.get("ov_barrier_timeout_seconds", 300)),
        noov_quiet_wait_seconds=int(runtime_raw.get("noov_quiet_wait_seconds", 3)),
        gateway_start_timeout_seconds=int(runtime_raw.get("gateway_start_timeout_seconds", 120)),
        gateway_health_poll_seconds=float(runtime_raw.get("gateway_health_poll_seconds", 2.0)),
        qa_retry_count=int(runtime_raw.get("qa_retry_count", 2)),
        qa_retry_backoff_seconds=float(runtime_raw.get("qa_retry_backoff_seconds", 2.0)),
        judge_parallel=int(runtime_raw.get("judge_parallel", 10)),
        strict_ov_usage=bool(int(os.environ.get("OV_OC_STRICT_OV_USAGE", str(int(runtime_raw.get("strict_ov_usage", True)))))),
        fallback_explicit_commit_for_telemetry=bool(int(os.environ.get(
            "OV_OC_FORCE_EXPLICIT_COMMIT_FOR_OV_TELEMETRY",
            str(int(runtime_raw.get("fallback_explicit_commit_for_telemetry", False))),
        ))),
        barrier_require_extracted_memories=bool(int(os.environ.get(
            "OV_OC_BARRIER_REQUIRE_MEMORIES",
            str(int(runtime_raw.get("barrier_require_extracted_memories", 0))),
        ))),
        run_ov_smoke_test=bool(int(os.environ.get("OV_OC_RUN_OV_SMOKE", str(int(runtime_raw.get("run_ov_smoke_test", True)))))),
        max_block_retries=int(runtime_raw.get("max_block_retries", 2)),
        openclaw_agent_id=str(runtime_raw.get("openclaw_agent_id", "main")),
        openviking_agent_id=str(runtime_raw.get("openviking_agent_id", "default")),
        openviking_port=int(runtime_raw.get("openviking_port", 1933)),
        gateway_base_url_fallback=str(runtime_raw.get("gateway_base_url_fallback", "http://127.0.0.1:18789")),
        telemetry_slop_seconds=float(runtime_raw.get("telemetry_slop_seconds", 1.0)),
        ingest_tail=str(runtime_raw.get("ingest_tail", "[remember what's said, keep existing memory]")),
    )

    repo_cfg = RepoConfig(**raw["repos"])
    version_cfg = VersionConfig(**raw["versions"])
    model_cfg = ModelEnvConfig(**raw["models"])
    env_cfg = EnvConfig(**raw["env"])

    dataset_raw = raw["dataset"]
    dataset_cfg = DatasetConfig(
        vendor_json=_as_path(root, dataset_raw["vendor_json"]),
        vendor_jsonl=_as_path(root, dataset_raw["vendor_jsonl"]),
        vendor_manifest=_as_path(root, dataset_raw["vendor_manifest"]),
        expected_total_cases=int(dataset_raw.get("expected_total_cases", 1540)),
        expected_sample_count=int(dataset_raw.get("expected_sample_count", 10)),
        expected_removed_category=int(dataset_raw.get("expected_removed_category", 5)),
    )

    groups = [GroupSpec(**g) for g in raw["groups"]]
    return ExperimentConfig(
        name=str(raw.get("name", "openviking-openclaw-benchmark")),
        repos=repo_cfg,
        runtime=runtime,
        versions=version_cfg,
        models=model_cfg,
        env=env_cfg,
        dataset=dataset_cfg,
        groups=groups,
    )
