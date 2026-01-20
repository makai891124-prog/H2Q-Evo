#!/usr/bin/env python3
"""H2Q-Evo v2.3.0 CLI - Main Entry Point"""

import sys
import click
from .commands import (
    InitCommand, ExecuteCommand, StatusCommand,
    ExportCommand, ImportCommand, VersionCommand
)


@click.group()
@click.version_option(version="2.3.0")
def main():
    """H2Q-Evo v2.3.0 - Autonomous Learning System CLI"""
    pass


@main.command("init")
def init_cmd():
    """Initialize H2Q-Evo agent"""
    cmd = InitCommand()
    cmd.run()


@main.command("execute")
@click.argument("task")
@click.option("--save-knowledge", is_flag=True, help="Save execution to knowledge base")
def execute_cmd(task, save_knowledge):
    """Execute a task"""
    cmd = ExecuteCommand()
    cmd.run(task=task, save_knowledge=save_knowledge)


@main.command("status")
def status_cmd():
    """Show agent status and knowledge statistics"""
    cmd = StatusCommand()
    cmd.run()


@main.command("export-checkpoint")
@click.argument("output_file")
def export_cmd(output_file):
    """Export agent checkpoint"""
    cmd = ExportCommand()
    cmd.run(output_file=output_file)


@main.command("import-checkpoint")
@click.argument("checkpoint_file")
def import_cmd(checkpoint_file):
    """Import agent checkpoint"""
    cmd = ImportCommand()
    cmd.run(checkpoint_file=checkpoint_file)


@main.command("version")
def version_cmd():
    """Show version information"""
    cmd = VersionCommand()
    cmd.run()


if __name__ == "__main__":
    main()
