"""The first 16 hex digits of the sha256 of one estimator (a file or a folder); our bench and submission tools use
the same definition, so a measurement record can be matched to the file that was uploaded."""
from __future__ import annotations

import hashlib
from pathlib import Path


def sha16(path: str | Path) -> str:
    p = Path(path).resolve()
    h = hashlib.sha256()
    if p.is_file():
        h.update(p.read_bytes())
    else:
        for f in sorted(q for q in p.rglob("*") if q.is_file() and "__pycache__" not in q.parts):
            h.update(f.relative_to(p).as_posix().encode("utf-8") + b"\0")
            h.update(f.read_bytes())
    return h.hexdigest()[:16]
