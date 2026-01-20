#!/usr/bin/env python3
"""H2Q-Evo Smoke Testing CLI"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from h2q_cli.commands import (
    InitCommand, ExecuteCommand, StatusCommand,
    ExportCommand, ImportCommand
)


def smoke_test():
    """Run smoke tests for CLI"""
    print("🔥 H2Q-Evo CLI Smoke Tests\n")
    
    try:
        # Test 1: Init
        print("[1/5] Testing Init Command...")
        init_cmd = InitCommand()
        init_cmd.run()
        print("✅ Init passed\n")
        
        # Test 2: Execute
        print("[2/5] Testing Execute Command...")
        exec_cmd = ExecuteCommand()
        exec_cmd.run(task="test", save_knowledge=True)
        print("✅ Execute passed\n")
        
        # Test 3: Status
        print("[3/5] Testing Status Command...")
        status_cmd = StatusCommand()
        status_cmd.run()
        print("✅ Status passed\n")
        
        # Test 4: Export
        print("[4/5] Testing Export Command...")
        export_cmd = ExportCommand()
        export_cmd.run(output_file="/tmp/test_ckpt.pkl")
        print("✅ Export passed\n")
        
        # Test 5: Import
        print("[5/5] Testing Import Command...")
        import_cmd = ImportCommand()
        import_cmd.run(checkpoint_file="/tmp/test_ckpt.pkl")
        print("✅ Import passed\n")
        
        print("✅ All smoke tests passed!")
        return 0
        
    except Exception as e:
        print(f"❌ Smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(smoke_test())
