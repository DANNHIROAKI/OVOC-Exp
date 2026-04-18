from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import ast
import hashlib
import json
import os
import re
import time
import venv

import requests

from .subproc import run_cmd
from .utils import deep_get, dump_json, ensure_dir, load_json, now_ms, write_text


def openviking_python_bin(venv_root: str | Path) -> Path:
    root = Path(venv_root)
    candidate = root / "bin" / "python"
    if candidate.exists():
        return candidate
    return root / "Scripts" / "python.exe"


def install_openviking(venv_root: str | Path, version: str, log_path: str | Path) -> Path:
    venv_root = Path(venv_root)
    py = openviking_python_bin(venv_root)
    if not py.exists():
        ensure_dir(venv_root.parent)
        builder = venv.EnvBuilder(with_pip=True, clear=False, symlinks=True, upgrade=False)
        builder.create(venv_root)
        py = openviking_python_bin(venv_root)
    run_cmd([str(py), "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"], check=True)
    run_cmd([str(py), "-m", "pip", "install", f"openviking=={version}"], check=True)
    actual = get_openviking_version(py)
    write_text(log_path, f"installed openviking=={actual}\n")
    if actual != version:
        raise RuntimeError(f"OpenViking version mismatch after install: expected {version}, got {actual}")
    return py


def get_openviking_version(python_bin: str | Path) -> str:
    code = "import importlib.metadata as m; print(m.version('openviking'))"
    result = run_cmd([str(python_bin), "-c", code], check=True)
    return result.stdout.strip()


def render_ov_conf(
    template_path: str | Path,
    output_path: str | Path,
    values: dict[str, Any],
    *,
    redact_api_keys: bool = False,
) -> Path:
    raw = Path(template_path).read_text(encoding="utf-8")
    for key, value in values.items():
        raw = raw.replace(f"__{key}__", str(value))
    if redact_api_keys:
        raw = re.sub(r'("api_key"\s*:\s*")[^"]+(")', r'\1<redacted>\2', raw)
        raw = re.sub(r'("root_api_key"\s*:\s*")[^"]+(")', r'\1<redacted>\2', raw)
    out = Path(output_path)
    ensure_dir(out.parent)
    out.write_text(raw, encoding="utf-8")
    try:
        os.chmod(out, 0o600)
    except OSError:
        pass
    return out


@dataclass(slots=True)
class OvUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    matched_events: int = 0
    source: str = "none"

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": int(self.input_tokens),
            "output_tokens": int(self.output_tokens),
            "total_tokens": int(self.total_tokens),
            "matched_events": int(self.matched_events),
            "source": self.source,
        }


@dataclass(slots=True)
class OvBarrierResult:
    session_id: str
    detail: dict[str, Any] | None
    context: dict[str, Any] | None
    wait_ms: int
    explicit_commit_payload: dict[str, Any] | None = None


class OpenVikingInspector:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        agent_id: str,
        *,
        account_id: str = "default",
        user_id: str = "default",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.agent_id = agent_id
        self.account_id = account_id
        self.user_id = user_id

    def headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        headers["X-OpenViking-Account"] = self.account_id
        headers["X-OpenViking-User"] = self.user_id
        if self.agent_id:
            headers["X-OpenViking-Agent"] = self.agent_id
        return headers

    def _request(
        self,
        path: str,
        *,
        method: str = "GET",
        json_body: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any] | list[Any] | None:
        url = self.base_url + path
        resp = requests.request(method, url, json=json_body, headers=self.headers(), timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, dict) and "result" in payload:
            return payload["result"]
        return payload

    def health(self) -> dict[str, Any] | None:
        resp = requests.get(self.base_url + "/health", timeout=15)
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, dict) else None

    def list_sessions(self) -> list[dict[str, Any]]:
        payload = self._request("/api/v1/sessions", timeout=20)
        return [x for x in payload or [] if isinstance(x, dict)]

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        payload = self._request(f"/api/v1/sessions/{session_id}", timeout=20)
        return payload if isinstance(payload, dict) else None

    def get_context(self, session_id: str, token_budget: int = 128000) -> dict[str, Any] | None:
        payload = self._request(f"/api/v1/sessions/{session_id}/context?token_budget={token_budget}", timeout=30)
        return payload if isinstance(payload, dict) else None

    def commit(self, session_id: str, *, wait: bool = True, telemetry: bool = True) -> dict[str, Any] | None:
        payload = self._request(
            f"/api/v1/sessions/{session_id}/commit",
            method="POST",
            json_body={"wait": wait, "telemetry": telemetry},
            timeout=120,
        )
        return payload if isinstance(payload, dict) else None

    def search_memories(self, query: str, *, limit: int = 5, telemetry: bool = True) -> dict[str, Any] | None:
        payload = self._request(
            "/api/v1/search/find",
            method="POST",
            json_body={
                "query": query,
                "target_uri": "viking://user/memories",
                "limit": limit,
                "telemetry": telemetry,
            },
            timeout=30,
        )
        return payload if isinstance(payload, dict) else None


def extract_memory_total(detail: dict[str, Any] | None) -> int:
    if not isinstance(detail, dict):
        return 0
    node = detail.get("memories_extracted")
    if not isinstance(node, dict):
        return 0
    if isinstance(node.get("total"), int):
        return int(node["total"])
    return sum(v for v in node.values() if isinstance(v, int))


def wait_for_commit_visibility(
    inspector: OpenVikingInspector,
    session_id: str,
    *,
    timeout_seconds: float,
    poll_seconds: float = 5.0,
    explicit_commit_fallback: bool = False,
    require_extracted_memories: bool = True,
) -> OvBarrierResult:
    started_ms = now_ms()
    deadline = time.time() + timeout_seconds
    detail: dict[str, Any] | None = None
    context: dict[str, Any] | None = None
    explicit_commit_payload: dict[str, Any] | None = None

    def _ready(detail: dict[str, Any] | None, context: dict[str, Any] | None) -> bool:
        commit_ok = isinstance(detail, dict) and isinstance(detail.get("commit_count"), int) and detail["commit_count"] > 0
        overview_ok = isinstance(context, dict) and isinstance(context.get("latest_archive_overview"), str) and bool(context["latest_archive_overview"].strip())
        memory_ok = extract_memory_total(detail) > 0
        if not require_extracted_memories:
            memory_ok = True
        return commit_ok and overview_ok and memory_ok

    while time.time() < deadline:
        detail = inspector.get_session(session_id)
        context = inspector.get_context(session_id)
        if _ready(detail, context):
            return OvBarrierResult(
                session_id=session_id,
                detail=detail,
                context=context,
                wait_ms=now_ms() - started_ms,
                explicit_commit_payload=explicit_commit_payload,
            )
        time.sleep(poll_seconds)

    if explicit_commit_fallback:
        explicit_commit_payload = inspector.commit(session_id, wait=True, telemetry=True)
        detail = inspector.get_session(session_id)
        context = inspector.get_context(session_id)
        if _ready(detail, context):
            return OvBarrierResult(
                session_id=session_id,
                detail=detail,
                context=context,
                wait_ms=now_ms() - started_ms,
                explicit_commit_payload=explicit_commit_payload,
            )

    return OvBarrierResult(
        session_id=session_id,
        detail=detail,
        context=context,
        wait_ms=now_ms() - started_ms,
        explicit_commit_payload=explicit_commit_payload,
    )


_TS_PATTERNS = [
    re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)"),
    re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:[\.,]\d+)?)"),
]


def _parse_ts_ms(line: str) -> int | None:
    for pattern in _TS_PATTERNS:
        match = pattern.search(line)
        if not match:
            continue
        text = match.group(1).replace(",", ".")
        try:
            if text.endswith("Z"):
                dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            elif "T" in text or "+" in text[10:] or "-" in text[10:]:
                dt = datetime.fromisoformat(text)
            else:
                dt = datetime.fromisoformat(text).replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except Exception:
            continue
    return None


def _extract_json_object(line: str) -> dict[str, Any] | None:
    if "{" not in line or "}" not in line:
        return None
    first = line.find("{")
    last = line.rfind("}")
    candidate = line[first:last + 1]
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    # OpenViking telemetry logs may render Python-literal dicts (single quotes),
    # which are not valid JSON but are safe to parse with literal_eval.
    try:
        obj = ast.literal_eval(candidate)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _first_present(mapping: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _coerce_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return 0


def _extract_usage_from_payload(payload: dict[str, Any]) -> OvUsage | None:
    candidates: list[dict[str, Any]] = []
    for key_path in (
        "telemetry.summary",
        "summary",
        "telemetry",
        "result.telemetry.summary",
        "result.summary",
        "usage",
    ):
        node = deep_get(payload, key_path)
        if isinstance(node, dict):
            candidates.append(node)
    if isinstance(payload, dict):
        candidates.append(payload)

    best: OvUsage | None = None
    for node in candidates:
        tokens_node = node.get("tokens") if isinstance(node, dict) else None
        if not isinstance(tokens_node, dict):
            usage_node = node.get("usage") if isinstance(node.get("usage"), dict) else None
            if usage_node:
                tokens_node = usage_node
        direct_in = _coerce_int(_first_present(node, ("input_tokens", "prompt_tokens")))
        direct_out = _coerce_int(_first_present(node, ("output_tokens", "completion_tokens")))
        direct_total = _coerce_int(_first_present(node, ("total_tokens",)))
        llm_in = llm_out = llm_total = emb_total = total = 0
        if isinstance(tokens_node, dict):
            llm = tokens_node.get("llm") if isinstance(tokens_node.get("llm"), dict) else {}
            emb = tokens_node.get("embedding") if isinstance(tokens_node.get("embedding"), dict) else {}
            llm_in = _coerce_int(_first_present(llm, ("input", "input_tokens")))
            llm_out = _coerce_int(_first_present(llm, ("output", "output_tokens")))
            llm_total = _coerce_int(_first_present(llm, ("total", "total_tokens")))
            emb_total = _coerce_int(_first_present(emb, ("total", "total_tokens", "input")))
            total = _coerce_int(_first_present(tokens_node, ("total", "total_tokens")))
        input_tokens = max(direct_in, llm_in + emb_total)
        output_tokens = max(direct_out, llm_out)
        total_tokens = max(direct_total, total, input_tokens + output_tokens, llm_total + emb_total)
        if total_tokens <= 0 and input_tokens <= 0 and output_tokens <= 0:
            continue
        candidate = OvUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            matched_events=1,
            source="telemetry",
        )
        if best is None or candidate.total_tokens > best.total_tokens:
            best = candidate
    return best


@dataclass(slots=True)
class OvLogEvent:
    ts_ms: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    session_id: str | None
    task_id: str | None
    source: str
    raw_hash: str


def parse_ov_log(log_path: str | Path) -> list[OvLogEvent]:
    path = Path(log_path)
    if not path.exists():
        return []
    events: list[OvLogEvent] = []
    seen: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        ts_ms = _parse_ts_ms(raw_line)
        if ts_ms is None:
            continue
        payload = _extract_json_object(raw_line)
        if not payload:
            continue
        usage = _extract_usage_from_payload(payload)
        if usage is None:
            continue
        raw_hash = hashlib.sha1(raw_line.encode("utf-8", errors="ignore")).hexdigest()
        if raw_hash in seen:
            continue
        seen.add(raw_hash)
        session_id = None
        task_id = None
        for key in ("session_id", "sessionId"):
            if isinstance(payload.get(key), str):
                session_id = payload[key]
                break
        for key in ("task_id", "taskId"):
            if isinstance(payload.get(key), str):
                task_id = payload[key]
                break
        events.append(
            OvLogEvent(
                ts_ms=ts_ms,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
                session_id=session_id,
                task_id=task_id,
                source="openviking.log",
                raw_hash=raw_hash,
            )
        )
    return events


def aggregate_usage_from_events(
    events: list[OvLogEvent],
    *,
    start_ms: int,
    end_ms: int,
    session_id: str | None = None,
    slop_seconds: float = 1.0,
) -> OvUsage:
    lo = start_ms - int(slop_seconds * 1000)
    hi = end_ms + int(slop_seconds * 1000)
    matched = [
        event for event in events
        if lo <= event.ts_ms <= hi and (session_id is None or event.session_id in {None, session_id})
    ]
    return OvUsage(
        input_tokens=sum(e.input_tokens for e in matched),
        output_tokens=sum(e.output_tokens for e in matched),
        total_tokens=sum(e.total_tokens for e in matched),
        matched_events=len(matched),
        source="openviking.log",
    )


def usage_from_payload(payload: dict[str, Any] | None, *, source: str) -> OvUsage:
    if not payload:
        return OvUsage(source=source)
    usage = _extract_usage_from_payload(payload) or OvUsage()
    usage.source = source
    return usage


def merge_usage(*usages: OvUsage) -> OvUsage:
    meaningful = [u for u in usages if u and (u.total_tokens or u.input_tokens or u.output_tokens or u.matched_events)]
    if not meaningful:
        return OvUsage(source="none")
    return OvUsage(
        input_tokens=sum(u.input_tokens for u in meaningful),
        output_tokens=sum(u.output_tokens for u in meaningful),
        total_tokens=sum(u.total_tokens for u in meaningful),
        matched_events=sum(u.matched_events for u in meaningful),
        source="+".join(u.source for u in meaningful),
    )
