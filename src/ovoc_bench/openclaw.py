from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import os
import re
import time

import json5
import requests

from .config import GroupSpec, RuntimeConfig
from .subproc import ManagedProcess, run_cmd
from .utils import (
    copy_file,
    dump_json,
    ensure_dir,
    load_json,
    mask_secret,
    now_ms,
    parse_boolish,
    recursive_find_first_http_url,
    recursive_find_port,
    redact_mapping,
    require_file,
    safe_unlink,
    utc_now_iso,
    write_text,
)


def openclaw_bin(prefix: str | Path) -> Path:
    return Path(prefix) / "bin" / "openclaw"


def install_openclaw(prefix: str | Path, version: str, log_path: str | Path) -> Path:
    prefix_p = Path(prefix)
    bin_path = openclaw_bin(prefix_p)
    if bin_path.exists():
        actual = get_openclaw_version(bin_path)
        if actual == version:
            return bin_path
    ensure_dir(prefix_p)
    cmd = (
        "set -euo pipefail; "
        f"curl -fsSL --proto '=https' --tlsv1.2 https://openclaw.ai/install-cli.sh | "
        f"bash -s -- --prefix {str(prefix_p)!s} --version {version}"
    )
    result = run_cmd(["bash", "-lc", cmd], check=True)
    write_text(log_path, result.stdout + "\n" + result.stderr)
    actual = get_openclaw_version(bin_path)
    if actual != version:
        raise RuntimeError(f"OpenClaw version mismatch after install: expected {version}, got {actual}")
    return bin_path


def get_openclaw_version(bin_path: str | Path) -> str:
    result = run_cmd([str(bin_path), "--version"], check=True)
    text = (result.stdout or result.stderr).strip()
    match = re.search(r"(\d{4}\.\d+\.\d+)", text)
    if match:
        return match.group(1)
    raise RuntimeError(f"Could not parse OpenClaw version from: {text}")


def write_openclaw_dotenv(home_root: str | Path, env_map: dict[str, str]) -> Path:
    home_root = Path(home_root)
    dot_env = home_root / ".openclaw" / ".env"
    ensure_dir(dot_env.parent)
    lines = [f"{k}={v}" for k, v in env_map.items()]
    dot_env.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dot_env


def onboard_custom_provider(
    bin_path: str | Path,
    *,
    env: dict[str, str],
    base_url: str,
    model_id: str,
    custom_compatibility: str = "openai",
) -> None:
    args = [
        str(bin_path),
        "onboard",
        "--non-interactive",
        "--mode",
        "local",
        "--skip-health",
        "--auth-choice",
        "custom-api-key",
        "--custom-base-url",
        base_url,
        "--custom-model-id",
        model_id,
        "--secret-input-mode",
        "ref",
        "--gateway-auth",
        "token",
        "--gateway-token-ref-env",
        "OPENCLAW_GATEWAY_TOKEN",
        "--accept-risk",
        "--custom-compatibility",
        custom_compatibility,
    ]
    run_cmd(args, env=env, check=True)


def install_plugin(bin_path: str | Path, plugin_dir: str | Path, *, env: dict[str, str]) -> None:
    run_cmd(
        [
            str(bin_path),
            "plugins",
            "install",
            str(plugin_dir),
            "--force",
            "--dangerously-force-unsafe-install",
        ],
        env=env,
        check=True,
    )
    run_cmd([str(bin_path), "plugins", "enable", "openviking"], env=env, check=False)


def config_set(bin_path: str | Path, path: str, value: Any, *, env: dict[str, str]) -> None:
    if isinstance(value, bool):
        rendered = "true" if value else "false"
    elif isinstance(value, (dict, list, int, float)):
        rendered = json.dumps(value, ensure_ascii=False)
    else:
        rendered = str(value)
    run_cmd([str(bin_path), "config", "set", path, rendered], env=env, check=True)


def config_get(bin_path: str | Path, path: str, *, env: dict[str, str]) -> Any:
    result = run_cmd([str(bin_path), "config", "get", path], env=env, check=False)
    if result.returncode != 0:
        return None
    text = result.stdout.strip()
    if text == "":
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    return text


def config_validate(bin_path: str | Path, *, env: dict[str, str]) -> None:
    run_cmd([str(bin_path), "config", "validate"], env=env, check=True)


def load_openclaw_config(config_path: str | Path) -> dict[str, Any]:
    text = Path(config_path).read_text(encoding="utf-8")
    data = json5.loads(text)
    if not isinstance(data, dict):
        raise ValueError("openclaw.json did not parse into a mapping.")
    return data


def save_redacted_openclaw_config(config_path: str | Path, output_path: str | Path) -> None:
    data = load_openclaw_config(config_path)
    dump_json(redact_mapping(data), output_path)


def apply_group_config(
    bin_path: str | Path,
    *,
    env: dict[str, str],
    runtime: RuntimeConfig,
    group: GroupSpec,
    ov_conf_path: str | Path,
) -> None:
    config_set(bin_path, "gateway.http.endpoints.responses.enabled", True, env=env)
    config_set(bin_path, "plugins.slots.memory", group.memory_slot, env=env)
    config_set(bin_path, "plugins.slots.contextEngine", group.context_engine, env=env)
    config_set(bin_path, "plugins.entries.openviking.enabled", group.openviking_enabled, env=env)
    config_set(bin_path, "plugins.deny", group.deny, env=env)
    config_set(bin_path, "gateway.mode", "local", env=env)
    if group.is_ov:
        config_set(bin_path, "plugins.entries.openviking.config.mode", "local", env=env)
        config_set(bin_path, "plugins.entries.openviking.config.configPath", str(ov_conf_path), env=env)
        config_set(bin_path, "plugins.entries.openviking.config.port", runtime.openviking_port, env=env)
        config_set(bin_path, "plugins.entries.openviking.config.agentId", runtime.openviking_agent_id, env=env)
        config_set(bin_path, "plugins.entries.openviking.config.autoCapture", True, env=env)
        config_set(bin_path, "plugins.entries.openviking.config.autoRecall", True, env=env)
        config_set(bin_path, "plugins.entries.openviking.config.emitStandardDiagnostics", True, env=env)
        config_set(bin_path, "plugins.entries.openviking.config.logFindRequests", True, env=env)
    config_validate(bin_path, env=env)


def inspect_plugin(bin_path: str | Path, *, env: dict[str, str]) -> str:
    result = run_cmd([str(bin_path), "plugins", "inspect", "openviking"], env=env, check=False)
    return (result.stdout or "") + (result.stderr or "")


def health_json(bin_path: str | Path, *, env: dict[str, str]) -> dict[str, Any] | None:
    result = run_cmd([str(bin_path), "health", "--json"], env=env, check=False)
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout.strip())
    except Exception:
        return None


def gateway_status_json(bin_path: str | Path, *, env: dict[str, str]) -> dict[str, Any] | None:
    result = run_cmd([str(bin_path), "gateway", "status", "--json"], env=env, check=False)
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout.strip())
    except Exception:
        return None


def infer_gateway_base_url(status_json: dict[str, Any] | None, fallback: str) -> str:
    if status_json:
        url = recursive_find_first_http_url(status_json)
        if url:
            if url.endswith("/"):
                return url.rstrip("/")
            return url
        port = recursive_find_port(status_json)
        if port:
            return f"http://127.0.0.1:{port}"
    return fallback.rstrip("/")


def start_gateway(bin_path: str | Path, *, env: dict[str, str], log_path: str | Path) -> ManagedProcess:
    proc = ManagedProcess([str(bin_path), "gateway", "run"], env=env, stdout_path=log_path)
    return proc.start()


def wait_for_gateway_ready(
    bin_path: str | Path,
    *,
    env: dict[str, str],
    timeout_seconds: float,
    poll_seconds: float,
    fallback_base_url: str,
) -> str:
    deadline = time.time() + timeout_seconds
    last_json: dict[str, Any] | None = None
    while time.time() < deadline:
        last_json = health_json(bin_path, env=env)
        if last_json is not None:
            status_json = gateway_status_json(bin_path, env=env)
            return infer_gateway_base_url(status_json, fallback_base_url)
        time.sleep(poll_seconds)
    raise RuntimeError(f"Gateway did not become healthy within {timeout_seconds} seconds. Last health payload: {last_json}")

def extract_response_text(body: dict[str, Any]) -> str:
    try:
        for item in body.get("output", []):
            if item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") in {"output_text", "text"}:
                        return content.get("text", "")
        for item in body.get("output", []):
            if "text" in item:
                return item["text"]
            for content in item.get("content", []):
                if "text" in content:
                    return content["text"]
    except Exception:
        pass
    return ""

def post_response(
    base_url: str,
    gateway_token: str,
    *,
    user: str,
    message: str,
    timeout_seconds: int,
    model: str = "openclaw",
    max_retries: int = 0,
    retry_backoff_seconds: float = 2.0,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/v1/responses"
    headers = {
        "Authorization": f"Bearer {gateway_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "input": message,
        "stream": False,
        "user": user,
    }
    attempt = 0
    while True:
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=timeout_seconds)
            if response.status_code in {429, 500, 502, 503, 504} and attempt < max_retries:
                attempt += 1
                time.sleep(retry_backoff_seconds * attempt)
                continue
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError):
            if attempt >= max_retries:
                raise
            attempt += 1
            time.sleep(retry_backoff_seconds * attempt)

def sessions_dir(openclaw_home: str | Path, agent_id: str) -> Path:
    return Path(openclaw_home) / "agents" / agent_id / "sessions"

def sessions_map_path(openclaw_home: str | Path, agent_id: str) -> Path:
    return sessions_dir(openclaw_home, agent_id) / "sessions.json"

def get_session_id_for_user(openclaw_home: str | Path, agent_id: str, user: str) -> str | None:
    path = sessions_map_path(openclaw_home, agent_id)
    if not path.exists():
        return None
    try:
        data = load_json(path)
    except Exception:
        return None
    candidate_keys = [
        f"agent:{agent_id}:openresponses-user:{user}",
        f"openresponses-user:{user}",
        user,
    ]
    for key in candidate_keys:
        item = data.get(key)
        if isinstance(item, dict) and isinstance(item.get("sessionId"), str):
            return item["sessionId"]
    # Fallback: choose the most recent session file.
    sess_dir = sessions_dir(openclaw_home, agent_id)
    jsonl_files = sorted(sess_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if jsonl_files:
        return jsonl_files[0].stem
    return None

def archive_session_file(openclaw_home: str | Path, agent_id: str, session_id: str) -> str | None:
    sess_dir = sessions_dir(openclaw_home, agent_id)
    src = sess_dir / f"{session_id}.jsonl"
    if not src.exists():
        return None
    dst = sess_dir / f"{session_id}.jsonl.{int(time.time())}"
    src.rename(dst)
    return dst.name

def read_runtime_config_summary(bin_path: str | Path, *, env: dict[str, str]) -> dict[str, Any]:
    paths = [
        "plugins.slots.memory",
        "plugins.slots.contextEngine",
        "plugins.entries.openviking.enabled",
        "plugins.deny",
        "gateway.http.endpoints.responses.enabled",
    ]
    summary: dict[str, Any] = {}
    for path in paths:
        summary[path] = config_get(bin_path, path, env=env)
    return summary

def assert_group_runtime(bin_path: str | Path, *, env: dict[str, str], group: GroupSpec) -> None:
    actual_context = config_get(bin_path, "plugins.slots.contextEngine", env=env)
    actual_memory = config_get(bin_path, "plugins.slots.memory", env=env)
    actual_enabled = config_get(bin_path, "plugins.entries.openviking.enabled", env=env)
    actual_deny = config_get(bin_path, "plugins.deny", env=env)
    if str(actual_context) != group.context_engine:
        raise RuntimeError(f"Context engine mismatch for {group.id}: expected {group.context_engine}, got {actual_context}")
    if str(actual_memory) != group.memory_slot:
        raise RuntimeError(f"Memory slot mismatch for {group.id}: expected {group.memory_slot}, got {actual_memory}")
    if parse_boolish(actual_enabled) != bool(group.openviking_enabled):
        raise RuntimeError(f"OpenViking enabled mismatch for {group.id}: expected {group.openviking_enabled}, got {actual_enabled}")
    actual_deny_list = actual_deny if isinstance(actual_deny, list) else [actual_deny] if actual_deny is not None else []
    if list(actual_deny_list) != list(group.deny):
        raise RuntimeError(f"plugins.deny mismatch for {group.id}: expected {group.deny}, got {actual_deny_list}")
