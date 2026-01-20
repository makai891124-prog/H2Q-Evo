"""Comprehensive test suite for H2Q-Evo v2.3.0 CLI and local agent."""
import json
import tempfile
from pathlib import Path

import pytest

from h2q_project.h2q_cli import commands, main
from h2q_project.local_executor import LocalExecutor
from h2q_project.knowledge.knowledge_db import KnowledgeDB
from h2q_project.monitoring.metrics_tracker import MetricsTracker
from h2q_project.persistence.checkpoint_manager import CheckpointManager


class TestLocalExecutor:
    def test_execute_basic(self):
        executor = LocalExecutor()
        result = executor.execute("What is 1+1?", strategy="default")
        assert "output" in result
        assert "confidence" in result
        assert result["elapsed_time"] >= 0

    def test_task_analysis(self):
        executor = LocalExecutor()
        task_info = executor._analyze_task("Calculate 2+2")
        assert task_info["type"] in ["math", "logic", "general"]
        assert "complexity" in task_info

    def test_task_classification(self):
        assert LocalExecutor._classify_task("Calculate 2+2") == "math"
        assert LocalExecutor._classify_task("Prove theorem") == "logic"
        assert LocalExecutor._classify_task("Random text") == "general"


class TestKnowledgeDB:
    def test_save_and_retrieve(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = KnowledgeDB(Path(tmpdir))
            experience = {
                "task": "test task",
                "task_type": "math",
                "result": {"output": "42"},
                "confidence": 0.8,
                "feedback": {"success": True},
                "timestamp": 0,
            }
            db.save_experience(experience)

            similar = db.retrieve_similar("test", top_k=1)
            assert len(similar) == 1
            assert similar[0]["task"] == "test task"

    def test_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = KnowledgeDB(Path(tmpdir))
            db.save_experience(
                {
                    "task": "task1",
                    "task_type": "math",
                    "result": {},
                    "confidence": 0.7,
                    "feedback": {},
                    "timestamp": 0,
                }
            )
            db.save_experience(
                {
                    "task": "task2",
                    "task_type": "logic",
                    "result": {},
                    "confidence": 0.6,
                    "feedback": {},
                    "timestamp": 0,
                }
            )
            stats = db.get_stats()
            assert stats["total_experiences"] == 2
            assert set(stats["domains"]) == {"math", "logic"}


class TestCheckpointManager:
    def test_checkpoint_create_and_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir)
            mgr = CheckpointManager()
            mgr.init(home)

            checkpoint = mgr.create_checkpoint(home)
            checkpoint_file = home / "test.ckpt"
            mgr.save(checkpoint, checkpoint_file)

            assert checkpoint_file.exists()
            loaded = mgr.load(checkpoint_file)
            assert loaded["version"] == "1.0.0"

    def test_checkpoint_verify(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir)
            mgr = CheckpointManager()
            mgr.init(home)

            checkpoint = mgr.create_checkpoint(home)
            checkpoint_file = home / "test.ckpt"
            mgr.save(checkpoint, checkpoint_file)

            assert mgr.verify_checkpoint(checkpoint_file)

    def test_checkpoint_restore(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            home_src = Path(tmpdir) / "src"
            home_dst = Path(tmpdir) / "dst"
            home_src.mkdir()

            (home_src / "config.json").write_text('{"version": "2.3.0"}')
            (home_src / "metrics.json").write_text('{"total_tasks": 5}')

            mgr = CheckpointManager()
            mgr.init(home_src)
            checkpoint = mgr.create_checkpoint(home_src)
            checkpoint_file = home_src / "test.ckpt"
            mgr.save(checkpoint, checkpoint_file)

            mgr.restore(checkpoint_file, home_dst)
            assert (home_dst / "config.json").exists()
            assert (home_dst / "metrics.json").exists()


class TestMetricsTracker:
    def test_record_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MetricsTracker(home=Path(tmpdir))
            result = {"confidence": 0.8, "output": "test"}
            tracker.record_execution("test task", result)

            metrics = tracker.get_current_metrics()
            assert metrics["total_tasks"] == 1
            assert metrics["success_rate"] > 0

    def test_success_rate_calculation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MetricsTracker(home=Path(tmpdir))
            tracker.record_execution("task1", {"confidence": 0.9})
            tracker.record_execution("task2", {"confidence": 0.1})

            metrics = tracker.get_current_metrics()
            assert 0 < metrics["success_rate"] < 1


class TestCLIIntegration:
    def test_init_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            old_home = os.environ.get("HOME")
            os.environ["H2Q_AGENT_HOME"] = tmpdir
            try:
                cmd = commands.InitCommand()
                cmd.run()
                assert (Path(tmpdir) / "knowledge").exists()
                assert (Path(tmpdir) / "checkpoints").exists()
                assert (Path(tmpdir) / "config.json").exists()
            finally:
                if old_home:
                    os.environ["HOME"] = old_home

    def test_execute_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            os.environ["H2Q_AGENT_HOME"] = tmpdir
            cmd = commands.ExecuteCommand(task="1+1", strategy="auto", save_knowledge=False)
            cmd.run()

    def test_status_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            os.environ["H2Q_AGENT_HOME"] = tmpdir
            cmd = commands.StatusCommand()
            cmd.run()

    def test_export_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            os.environ["H2Q_AGENT_HOME"] = tmpdir
            init_cmd = commands.InitCommand()
            init_cmd.run()

            checkpoint_path = Path(tmpdir) / "export.ckpt"
            cmd = commands.ExportCommand(output=str(checkpoint_path))
            cmd.run()
            assert checkpoint_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
