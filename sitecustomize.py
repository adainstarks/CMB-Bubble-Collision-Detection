"""Make organized script folders importable from repository-root commands."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SCRIPTS_ROOT = REPO_ROOT / "scripts"

if SCRIPTS_ROOT.is_dir():
    paths = [SCRIPTS_ROOT]
    paths.extend(
        sorted(
            path
            for path in SCRIPTS_ROOT.iterdir()
            if path.is_dir() and path.name != "__pycache__"
        )
    )
    for path in reversed(paths):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)
