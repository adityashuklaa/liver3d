"""File checksums used for audit and reproducibility records."""
from __future__ import annotations

import hashlib


def file_sha256(path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def short(checksum: str, size: int = 12) -> str:
    return checksum[:size]
