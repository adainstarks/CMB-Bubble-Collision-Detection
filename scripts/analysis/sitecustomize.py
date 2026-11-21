"""Make repository script folders importable when running nested CLIs."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
PATHS = [SCRIPTS_ROOT]
PATHS.extend(
    sorted(
        path
        for path in SCRIPTS_ROOT.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    )
)
for path in reversed(PATHS):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)
