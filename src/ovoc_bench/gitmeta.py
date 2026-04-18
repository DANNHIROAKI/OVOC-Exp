from __future__ import annotations

from pathlib import Path
import json

from .subproc import run_cmd
from .utils import ensure_dir


def clone_or_update(url: str, dest: str | Path, *, branch: str | None = None) -> Path:
    dest_p = Path(dest)
    ensure_dir(dest_p.parent)
    if not dest_p.exists():
        args = ["git", "clone", "--depth", "1"]
        if branch:
            args.extend(["--branch", branch])
        args.extend([url, str(dest_p)])
        run_cmd(args, check=True)
    else:
        run_cmd(["git", "-C", str(dest_p), "fetch", "--depth", "1", "origin"], check=True)
        if branch:
            run_cmd(["git", "-C", str(dest_p), "checkout", branch], check=True)
            run_cmd(["git", "-C", str(dest_p), "pull", "--ff-only", "origin", branch], check=True)
        else:
            run_cmd(["git", "-C", str(dest_p), "pull", "--ff-only"], check=False)
    return dest_p


def get_commit_sha(repo_dir: str | Path) -> str | None:
    repo_p = Path(repo_dir)
    if not repo_p.exists():
        return None
    result = run_cmd(["git", "-C", str(repo_p), "rev-parse", "HEAD"], check=False)
    if result.returncode == 0:
        return result.stdout.strip()
    return None


def get_remote_url(repo_dir: str | Path) -> str | None:
    result = run_cmd(["git", "-C", str(repo_dir), "remote", "get-url", "origin"], check=False)
    if result.returncode == 0:
        return result.stdout.strip()
    return None
