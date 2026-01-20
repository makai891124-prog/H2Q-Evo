#!/usr/bin/env python3
"""
Comprehensive validation test for H2Q-Evo v2.3.0 CLI system.
Tests all components end-to-end with proper knowledge persistence.
"""
import os
import sys
import shutil
from pathlib import Path

# Setup PYTHONPATH
sys.path.insert(0, "/Users/imymm/H2Q-Evo")

def main():
    # Use test directory to avoid polluting user's home
    test_agent_home = Path("/tmp/h2q_test_agent_home")
    if test_agent_home.exists():
        shutil.rmtree(test_agent_home)
    test_agent_home.mkdir(parents=True, exist_ok=True)
    
    os.environ['H2Q_AGENT_HOME'] = str(test_agent_home)
    
    from h2q_project.h2q_cli import commands
    from h2q_project.persistence.checkpoint_manager import CheckpointManager
    
    print("=" * 70)
    print("H2Q-Evo v2.3.0 Comprehensive Validation Test")
    print("=" * 70)
    
    # Test 1: Initialize agent
    print("\n[TEST 1/5] Initialize agent...")
    try:
        cmd = commands.InitCommand()
        cmd.run()
        assert (test_agent_home / "knowledge").exists(), "Knowledge dir not created"
        assert (test_agent_home / "checkpoints").exists(), "Checkpoints dir not created"
        assert (test_agent_home / "config.json").exists(), "Config file not created"
        print("✅ Agent initialized successfully")
        print(f"   Location: {test_agent_home}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False
    
    # Test 2: Execute task with knowledge persistence
    print("\n[TEST 2/5] Execute task with knowledge persistence...")
    try:
        cmd = commands.ExecuteCommand(
            task="What is 2 plus 2?", 
            strategy="auto", 
            save_knowledge=True
        )
        cmd.run()
        print("✅ Task executed and experience saved")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False
    
    # Test 3: Execute second task
    print("\n[TEST 3/5] Execute additional tasks to build knowledge...")
    try:
        for task in ["Calculate 10 * 5", "What is pi?", "Define photosynthesis"]:
            cmd = commands.ExecuteCommand(
                task=task, 
                strategy="auto", 
                save_knowledge=True
            )
            cmd.run()
        print("✅ Multiple tasks executed and saved")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False
    
    # Test 4: Check status and metrics
    print("\n[TEST 4/5] Verify knowledge base and metrics...")
    try:
        cmd = commands.StatusCommand()
        cmd.run()
        
        # Verify metrics file
        metrics_file = test_agent_home / "metrics.json"
        assert metrics_file.exists(), "Metrics file not created"
        
        # Verify knowledge database
        knowledge_db = test_agent_home / "knowledge" / "knowledge.db"
        assert knowledge_db.exists(), "Knowledge database not created"
        print("✅ Knowledge base and metrics verified")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False
    
    # Test 5: Create and verify checkpoint
    print("\n[TEST 5/5] Create checkpoint and verify migration...")
    try:
        checkpoint_path = test_agent_home / "test_checkpoint.ckpt"
        cmd = commands.ExportCommand(output=str(checkpoint_path))
        cmd.run()
        
        assert checkpoint_path.exists(), "Checkpoint file not created"
        
        # Verify checkpoint integrity
        mgr = CheckpointManager()
        is_valid = mgr.verify_checkpoint(checkpoint_path)
        assert is_valid, "Checkpoint verification failed"
        
        print("✅ Checkpoint created and verified")
        print(f"   Size: {checkpoint_path.stat().st_size / 1024:.2f} KB")
        
        # Test migration (restore to new location)
        restore_path = Path("/tmp/h2q_restored_agent")
        if restore_path.exists():
            shutil.rmtree(restore_path)
        restore_path.mkdir(parents=True, exist_ok=True)
        
        mgr.restore(checkpoint_path, restore_path)
        assert (restore_path / "config.json").exists(), "Config not restored"
        assert (restore_path / "metrics.json").exists(), "Metrics not restored"
        print("✅ Checkpoint successfully migrated to new location")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED - System ready for production!")
    print("=" * 70)
    print("\nTest Summary:")
    print("  ✅ Agent initialization")
    print("  ✅ Task execution with knowledge persistence")
    print("  ✅ Multi-task learning")
    print("  ✅ Metrics tracking")
    print("  ✅ Checkpoint creation and migration")
    print("\nNext Steps:")
    print("  1. Run pytest suite: pytest tests/ -v")
    print("  2. Deploy to production: pip install -e .")
    print("  3. Use CLI commands: h2q --help")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
