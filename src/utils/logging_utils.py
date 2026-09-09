"""Console + JSONL run logging. Records must never contain identifiers."""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, Dict

_FORMAT = "%(asctime)s %(levelname)s %(name)s :: %(message)s"


def get_logger(name: str = "liver3d", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


class JsonlLogger:
    """Append-only structured log for training and evaluation events."""

    def __init__(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path

    def log(self, **record: Any) -> Dict[str, Any]:
        record.setdefault("ts", time.time())
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        return record
