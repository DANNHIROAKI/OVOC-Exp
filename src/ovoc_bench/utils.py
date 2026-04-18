from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import hashlib
import json
import os
import random
import socket
import re
import shutil
import string
import time


SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|token|password|secret)"),
)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def now_ms() -> int:
    return int(time.time() * 1000)


def now_s() -> float:
    return time.time()


def sleep_seconds(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def find_free_loopback_port() -> int:
    """Return an ephemeral free TCP port on 127.0.0.1 (for isolated gateway binds)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_text(path: str | Path, text: str) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(text, encoding="utf-8")
    return p


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def dump_json(obj: Any, path: str | Path, *, indent: int = 2) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, ensure_ascii=False, sort_keys=False)
    return p


def append_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return p


def dump_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return p


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_json_sha1(obj: Any) -> str:
    data = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(data).hexdigest()


def copytree(src: str | Path, dst: str | Path, *, dirs_exist_ok: bool = False) -> Path:
    src_p = Path(src)
    dst_p = Path(dst)
    ensure_dir(dst_p.parent)
    shutil.copytree(src_p, dst_p, dirs_exist_ok=dirs_exist_ok)
    return dst_p


def rm_tree(path: str | Path) -> None:
    p = Path(path)
    if p.exists():
        shutil.rmtree(p)


def copy_file(src: str | Path, dst: str | Path) -> Path:
    src_p = Path(src)
    dst_p = Path(dst)
    ensure_dir(dst_p.parent)
    shutil.copy2(src_p, dst_p)
    return dst_p


def random_token(nbytes: int = 32) -> str:
    return hashlib.sha256(os.urandom(nbytes)).hexdigest()


def short_uid(prefix: str = "", k: int = 10) -> str:
    alphabet = string.ascii_lowercase + string.digits
    tail = "".join(random.choice(alphabet) for _ in range(k))
    return f"{prefix}{tail}"


def slugify(text: str) -> str:
    out = text.lower().strip()
    out = re.sub(r"[^a-z0-9._-]+", "-", out)
    out = re.sub(r"-{2,}", "-", out).strip("-")
    return out or "item"


def normalize_answer_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def parse_boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def load_dotenv(path: str | Path) -> dict[str, str]:
    p = Path(path)
    env: dict[str, str] = {}
    if not p.exists():
        return env
    for raw_line in p.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        env[key.strip()] = val.strip().strip('"').strip("'")
    return env


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable is missing: {name}")
    return value


def resolve_any_env(*names: str) -> tuple[str, str]:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return name, value
    raise RuntimeError(f"None of the expected environment variables is set: {', '.join(names)}")


def deep_get(obj: Any, path: str, default: Any = None) -> Any:
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def recursive_find_strings(obj: Any) -> list[str]:
    values: list[str] = []
    if isinstance(obj, dict):
        for value in obj.values():
            values.extend(recursive_find_strings(value))
    elif isinstance(obj, list):
        for value in obj:
            values.extend(recursive_find_strings(value))
    elif isinstance(obj, str):
        values.append(obj)
    return values


def recursive_find_first_http_url(obj: Any) -> str | None:
    for text in recursive_find_strings(obj):
        if text.startswith("http://") or text.startswith("https://"):
            return text
    return None


def recursive_find_port(obj: Any) -> int | None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, int) and "port" in key.lower():
                return value
            found = recursive_find_port(value)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = recursive_find_port(item)
            if found is not None:
                return found
    return None


def dataclass_to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    return obj


def mask_secret(value: str, *, keep: int = 4) -> str:
    if not value:
        return value
    if len(value) <= keep * 2:
        return "*" * len(value)
    return f"{value[:keep]}***{value[-keep:]}"


def redact_value(key: str, value: Any) -> Any:
    key_l = key.lower()
    if any(p.search(key_l) for p in SECRET_PATTERNS):
        if isinstance(value, str):
            return mask_secret(value)
        return "<redacted>"
    if isinstance(value, dict):
        return {k: redact_value(k, v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_value(key, v) for v in value]
    return value


def redact_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    return {k: redact_value(k, v) for k, v in mapping.items()}


def require_file(path: str | Path) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p


def safe_unlink(path: str | Path) -> None:
    p = Path(path)
    if p.exists():
        p.unlink()


def human_ms(ms: float | int) -> str:
    return f"{float(ms)/1000:.2f}s"


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out
