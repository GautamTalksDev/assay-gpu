"""Command-line entrypoint for assay."""

from __future__ import annotations

import typer

app = typer.Typer(no_args_is_help=True, help="Deterministic GPU correctness assay.")


@app.callback()
def main() -> None:
    """Deterministic GPU correctness assay."""


@app.command()
def run() -> None:
    """Run the assay. Detector logic is not implemented in this checkpoint."""
