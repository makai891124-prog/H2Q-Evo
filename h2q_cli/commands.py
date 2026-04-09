"""H2Q-Evo v2.3.0 CLI Commands Implementation"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from h2q_project.local_executor import LocalExecutor
from h2q_project.monitoring.metrics_tracker import MetricsTracker
from h2q_project.persistence.checkpoint_manager import CheckpointManager


class BaseCommand:
    """Base command with common functionality"""
    
    def __init__(self):
        self.home = self.agent_home()
        self.executor = LocalExecutor()
        self.metrics = MetricsTracker(home=self.home)
        
    @staticmethod
    def agent_home() -> Path:
        """Get agent home directory with env var support"""
        env_home = os.environ.get("H2Q_AGENT_HOME")
        if env_home:
            return Path(env_home)
        return Path.home() / ".h2q-evo"


class InitCommand(BaseCommand):
    """Initialize agent"""
    
    def run(self):
        """Initialize agent home directory"""
        self.home.mkdir(parents=True, exist_ok=True)
        (self.home / "knowledge").mkdir(exist_ok=True)
        (self.home / "metrics.json").touch()
        (self.home / "config.json").touch()
        
        config = {
            "version": "2.3.0",
            "created_at": datetime.now().isoformat(),
            "inference_mode": "local"
        }
        (self.home / "config.json").write_text(json.dumps(config, indent=2))
        print(f"✅ Agent initialized at {self.home}")


class ExecuteCommand(BaseCommand):
    """Execute a task"""
    
    def run(self, task: str = "", save_knowledge: bool = False):
        """Execute task and optionally save to knowledge base"""
        self.executor.init_knowledge_db(self.home)
        
        result = self.executor.execute(task)
        print(f"\n📝 Task: {task}")
        print(f"✅ Output: {result['output']}")
        print(f"🎯 Confidence: {result['confidence']:.2f}")
        print(f"⏱️  Time: {result['elapsed_time']:.2f}s")
        
        if save_knowledge and self.executor.knowledge_db:
            self.executor.save_experience(
                task=task,
                result=result,
                feedback={"user_confirmed": True},
            )
            print("💾 Experience saved to knowledge base")
 copilot/analyze-cli-error-reasons


        
 main
        self.metrics.record_execution(task, result)


class StatusCommand(BaseCommand):
    """Show agent status"""
    
    def run(self):
        """Display agent status and knowledge statistics"""
        self.executor.init_knowledge_db(self.home)
        
        print(f"\n🤖 Agent Status:")
        print(f"   Home: {self.home}")
        print(f"   Version: 2.3.0")
        
        stats = self.executor.get_knowledge_stats(self.home)
        print(f"\n📚 Knowledge Base:")
        print(f"   Total Experiences: {stats.get('total_experiences', 0)}")
        print(f"   Domains: {stats.get('domains', 'N/A')}")
        
        if (self.home / "metrics.json").exists():
            metrics = json.loads((self.home / "metrics.json").read_text() or "{}")
            print(f"\n📊 Metrics:")
            print(f"   Total Tasks: {metrics.get('total_tasks', 0)}")
            print(f"   Success Rate: {metrics.get('success_rate', 0):.2%}")


class ExportCommand(BaseCommand):
    """Export checkpoint"""
    
    def run(self, output_file: str = ""):
        """Export agent checkpoint"""
 copilot/analyze-cli-error-reasons
        if not output_file:
            print("❌ Export failed: output_file must be provided")
            return
        mgr = CheckpointManager()


        mgr = CheckpointManager()
        mgr.init(self.home)
        
 main
        try:
            checkpoint = mgr.create_checkpoint(self.home)
            mgr.save(checkpoint, Path(output_file))
            print(f"✅ Checkpoint exported to {output_file}")
        except Exception as e:
            print(f"❌ Export failed: {e}")


class ImportCommand(BaseCommand):
    """Import checkpoint"""
    
    def run(self, checkpoint_file: str = ""):
        """Import agent checkpoint"""
 copilot/analyze-cli-error-reasons
        if not checkpoint_file:
            print("❌ Import failed: checkpoint_file must be provided")
            return
        mgr = CheckpointManager()


        mgr = CheckpointManager()
        
 main
        try:
            mgr.restore(Path(checkpoint_file), self.home)
            print(f"✅ Checkpoint imported from {checkpoint_file}")
        except Exception as e:
            print(f"❌ Import failed: {e}")


class VersionCommand(BaseCommand):
    """Show version"""
    
    def run(self):
        """Display version information"""
        print("\n🎯 H2Q-Evo v2.3.0")
        print("   Type: Local Learning System")
        print("   Status: Production Ready ✅")
        print("   License: MIT")
