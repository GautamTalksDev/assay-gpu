"""Run a user script after hooks are installed. No network. No code rewrite."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def run_user_script(argv: list[str]) -> None:
    """Execute argv[0] as __main__ with remaining argv. In-process so hooks apply."""
    if not argv:
        msg = "pass a script path after assay watch options"
        raise ValueError(msg)
    script = Path(argv[0])
    if not script.is_file():
        msg = f"script not found: {script}"
        raise FileNotFoundError(msg)
    sys.argv = argv
    runpy.run_path(str(script), run_name="__main__")
