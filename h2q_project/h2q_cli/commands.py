"""Command implementations for the H2Q-Evo CLI."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

from ..local_executor import LocalExecutor
from ..monitoring.metrics_tracker import MetricsTracker
from ..persistence.checkpoint_manager import CheckpointManager


class BaseCommand:
    """Shared helpers for CLI commands."""

    def __init__(self) -> None:
        self.home = self.agent_home()  # Get home early
        self.executor = LocalExecutor()
        self.checkpoint_mgr = CheckpointManager()
        self.metrics = MetricsTracker(home=self.home)  # Pass home to metrics tracker

    @staticmethod
    def agent_home() -> Path:
        # Check for H2Q_AGENT_HOME environment variable first
        env_home = os.environ.get("H2Q_AGENT_HOME")
        if env_home:
            home = Path(env_home).expanduser()
        else:
            home = Path.home() / ".h2q-evo"
        home.mkdir(parents=True, exist_ok=True)
        return home

    def run(self) -> None:  # pragma: no cover - interface only
        raise NotImplementedError


class InitCommand(BaseCommand):
    def run(self) -> None:
        print("Initializing H2Q-Evo agent...")

        (self.home / "knowledge").mkdir(exist_ok=True)
        (self.home / "checkpoints").mkdir(exist_ok=True)
        (self.home / "logs").mkdir(exist_ok=True)

        self.executor.init_knowledge_db(self.home)
        self.checkpoint_mgr.init(self.home)

        config = {
            "version": "2.3.0",
            "home": str(self.home),
        }
        with (self.home / "config.json").open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        print(f"Agent home: {self.home}")
        print("Done.")


class ExecuteCommand(BaseCommand):
    def __init__(self, task: str, strategy: str, save_knowledge: bool) -> None:
        super().__init__()
        self.task = task
        self.strategy = strategy
        self.save_knowledge = save_knowledge

    def run(self) -> None:
        # Initialize executor with home directory
        self.executor.init_knowledge_db(self.home)
        
        print(f"Task: {self.task}")
        result = self.executor.execute(self.task, strategy=self.strategy)

        print("Result:\n" + result.get("output", ""))
        print(f"Confidence: {result.get('confidence', 0.0):.2f}")
        print(f"Elapsed: {result.get('elapsed_time', 0.0):.2f}s")

        if self.save_knowledge:
            self.executor.save_experience(task=self.task, result=result, feedback={"user_confirmed": True})
            print("Experience saved.")

        self.metrics.record_execution(self.task, result)


class StatusCommand(BaseCommand):
    def run(self) -> None:
        self.executor.init_knowledge_db(self.home)
        stats = self.executor.get_knowledge_stats(self.home)
        metrics = self.metrics.get_current_metrics()

        print("Knowledge base:")
        print(f"  total_experiences: {stats.get('total_experiences', 0)}")
        print(f"  domains: {', '.join(stats.get('domains', []))}")

        print("Metrics:")
        print(f"  total_tasks: {metrics.get('total_tasks', 0)}")
        print(f"  success_rate: {metrics.get('success_rate', 0.0):.2f}")


class ExportCommand(BaseCommand):
    def __init__(self, output: str) -> None:
        super().__init__()
        self.output = Path(output)

    def run(self) -> None:
        checkpoint = self.checkpoint_mgr.create_checkpoint(self.home)
        self.checkpoint_mgr.save(checkpoint, self.output)
        checksum = self.checkpoint_mgr.compute_checksum(self.output)

        print(f"Checkpoint exported: {self.output}")
        print(f"Checksum: {checksum[:12]}...")


class ImportCommand(BaseCommand):
    def __init__(self, checkpoint: str) -> None:
        super().__init__()
        self.checkpoint = Path(checkpoint)

    def run(self) -> None:
        if not self.checkpoint_mgr.verify_checkpoint(self.checkpoint):
            print("Checkpoint verification failed.")
            return

        home = self.agent_home()
        self.checkpoint_mgr.restore(self.checkpoint, home)
        print("Checkpoint imported.")
