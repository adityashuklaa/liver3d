"""Environment capture for the experiment record."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from typing import Any, Dict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=_REPO_ROOT,
        )
        return out.stdout.strip() or "not_a_git_repo"
    except Exception:  # pragma: no cover - git optional
        return "unavailable"


def environment_record() -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "code_commit": _git_commit(),
    }
    try:
        import torch

        record["torch"] = torch.__version__
        record["cuda_available"] = bool(torch.cuda.is_available())
        record["cuda"] = torch.version.cuda
        record["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:  # pragma: no cover
        record["torch"] = None
    for mod in ("numpy", "scipy", "nibabel"):
        try:
            record[mod] = __import__(mod).__version__
        except ImportError:  # pragma: no cover
            record[mod] = None
    return record


def write_environment(path: str) -> Dict[str, Any]:
    rec = environment_record()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=2)
    return rec
