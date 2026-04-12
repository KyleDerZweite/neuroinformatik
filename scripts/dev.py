#!/usr/bin/env python3
"""Development entry point that starts both API and web apps."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT_DIR / "src" / "web"

API_HOST = os.environ.get("API_HOST", "127.0.0.1")
API_PORT = os.environ.get("API_PORT", "8000")
WEB_HOST = os.environ.get("WEB_HOST", "127.0.0.1")
WEB_PORT = os.environ.get("WEB_PORT", "5173")


def _require_path(path: Path, message: str) -> None:
    if not path.exists():
        raise SystemExit(message)


def _spawn(command: list[str], cwd: Path) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        command,
        cwd=cwd,
        start_new_session=True,
    )


def _terminate(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def main() -> int:
    _require_path(ROOT_DIR / "pyproject.toml", "Missing pyproject.toml at repository root.")
    _require_path(WEB_DIR / "package.json", "Missing src/web/package.json.")
    _require_path(ROOT_DIR / "uv.lock", "Missing uv.lock. Run `uv lock` first.")
    _require_path(WEB_DIR / "pnpm-lock.yaml", "Missing src/web/pnpm-lock.yaml. Run `pnpm install` in src/web first.")
    _require_path(WEB_DIR / "node_modules", "Missing src/web/node_modules. Run `pnpm install` in src/web first.")

    api_command = [
        "uv",
        "run",
        "uvicorn",
        "src.api.main:app",
        "--reload",
        "--host",
        API_HOST,
        "--port",
        API_PORT,
    ]
    web_command = [
        "pnpm",
        "dev",
        "--host",
        WEB_HOST,
        "--port",
        WEB_PORT,
    ]

    print(f"API: http://{API_HOST}:{API_PORT}")
    print(f"Web: http://{WEB_HOST}:{WEB_PORT}")

    processes = [
        _spawn(api_command, ROOT_DIR),
        _spawn(web_command, WEB_DIR),
    ]

    try:
        while True:
            for process in processes:
                return_code = process.poll()
                if return_code is not None:
                    return return_code
            time.sleep(0.25)
    except KeyboardInterrupt:
        return 0
    finally:
        for process in reversed(processes):
            _terminate(process)


if __name__ == "__main__":
    sys.exit(main())
