from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import os
import shlex
import signal
import subprocess
import time

import psutil

from .utils import ensure_dir


@dataclass(slots=True)
class CmdResult:
    args: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def raise_for_status(self) -> None:
        if self.returncode != 0:
            rendered = " ".join(shlex.quote(a) for a in self.args)
            raise RuntimeError(
                f"Command failed with exit {self.returncode}: {rendered}\nSTDOUT:\n{self.stdout}\nSTDERR:\n{self.stderr}"
            )


def run_cmd(
    args: Sequence[str],
    *,
    env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
    timeout: float | None = None,
    check: bool = True,
    text: bool = True,
) -> CmdResult:
    proc = subprocess.run(
        list(args),
        env=env,
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=text,
        timeout=timeout,
    )
    result = CmdResult(list(args), proc.returncode, proc.stdout or "", proc.stderr or "")
    if check:
        result.raise_for_status()
    return result


class ManagedProcess:
    def __init__(
        self,
        args: Sequence[str],
        *,
        env: dict[str, str] | None = None,
        cwd: str | Path | None = None,
        stdout_path: str | Path | None = None,
        stderr_to_stdout: bool = True,
    ) -> None:
        self.args = list(args)
        self.env = env
        self.cwd = str(cwd) if cwd is not None else None
        self.stdout_path = Path(stdout_path) if stdout_path is not None else None
        self.stderr_to_stdout = stderr_to_stdout
        self._fh: subprocess.Popen | None = None
        self._log_handle = None

    def start(self) -> "ManagedProcess":
        stdout_target = subprocess.PIPE
        stderr_target = subprocess.PIPE
        if self.stdout_path is not None:
            ensure_dir(self.stdout_path.parent)
            self._log_handle = self.stdout_path.open("a", encoding="utf-8")
            stdout_target = self._log_handle
            stderr_target = subprocess.STDOUT if self.stderr_to_stdout else self._log_handle
        self._fh = subprocess.Popen(
            self.args,
            env=self.env,
            cwd=self.cwd,
            stdout=stdout_target,
            stderr=stderr_target,
            text=True,
            start_new_session=True,
        )
        return self

    @property
    def pid(self) -> int | None:
        return self._fh.pid if self._fh else None

    def poll(self) -> int | None:
        if not self._fh:
            return None
        return self._fh.poll()

    def terminate_tree(self, *, timeout: float = 20.0) -> None:
        if not self._fh:
            return
        try:
            root = psutil.Process(self._fh.pid)
        except psutil.Error:
            root = None
        procs = [root] if root else []
        if root:
            try:
                procs.extend(root.children(recursive=True))
            except psutil.Error:
                pass
        for proc in reversed([p for p in procs if p is not None]):
            try:
                proc.terminate()
            except psutil.Error:
                pass
        deadline = time.time() + timeout
        alive = [p for p in procs if p is not None]
        while alive and time.time() < deadline:
            alive = [p for p in alive if p.is_running()]
            time.sleep(0.25)
        for proc in alive:
            try:
                proc.kill()
            except psutil.Error:
                pass
        if self._log_handle is not None:
            self._log_handle.flush()
            self._log_handle.close()
            self._log_handle = None

    def wait(self, timeout: float | None = None) -> int:
        if not self._fh:
            raise RuntimeError("Process has not been started.")
        return self._fh.wait(timeout=timeout)

    def __enter__(self) -> "ManagedProcess":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.terminate_tree()
