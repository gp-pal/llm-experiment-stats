"""The recorded replies: one JSONL file per arm, one line per item.

The runner only appends to these files and the report only reads them, so this is
the whole interface between calling a model and analysing what it said.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

Record = dict[str, Any]


def run_path(runs_dir: Path, arm: str) -> Path:
    return runs_dir / f"{arm}.jsonl"


def read_records(path: Path) -> list[Record]:
    """Every record in the file, in the order written. A missing file has none."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]
