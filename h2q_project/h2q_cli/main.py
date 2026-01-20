"""Entry point for the H2Q-Evo CLI."""
from __future__ import annotations

import sys
import click

from . import commands


@click.group()
@click.version_option(version="2.3.0")
def cli() -> None:
    """H2Q-Evo local agent CLI."""


@cli.command()
def init() -> None:
    """Initialize local agent state."""
    commands.InitCommand().run()


@cli.command()
@click.argument("task", type=str)
@click.option("--strategy", default="auto", help="Execution strategy")
@click.option("--save-knowledge", is_flag=True, help="Persist experience after run")
def execute(task: str, strategy: str, save_knowledge: bool) -> None:
    """Execute a task locally."""
    commands.ExecuteCommand(task=task, strategy=strategy, save_knowledge=save_knowledge).run()


@cli.command()
def status() -> None:
    """Show agent status and metrics."""
    commands.StatusCommand().run()


@cli.command(name="export-checkpoint")
@click.argument("output", type=click.Path())
def export_checkpoint(output: str) -> None:
    """Export agent checkpoint."""
    commands.ExportCommand(output=output).run()


@cli.command(name="import-checkpoint")
@click.argument("checkpoint", type=click.Path(exists=True))
def import_checkpoint(checkpoint: str) -> None:
    """Import agent checkpoint."""
    commands.ImportCommand(checkpoint=checkpoint).run()


def main() -> None:
    try:
        cli()  # pragma: no cover - delegated to click
    except KeyboardInterrupt:
        sys.exit(1)


if __name__ == "__main__":
    main()
