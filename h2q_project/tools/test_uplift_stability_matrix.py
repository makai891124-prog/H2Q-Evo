#!/usr/bin/env python3
"""
Phase 1 Track A - Test Matrix for Uplift Stability
Tests the uplift window tracker and slope-alarm mechanism across 3 duration tiers.

Targets:
- 12-cycle smoke (≤15min): uplift ≥ -0.010
- 24-cycle balanced (≤45min): uplift ≥ -0.005
- 48-cycle endurance (≤3hr): uplift ≥ 0.000
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_test_iteration(
    cycle_count: int,
    test_name: str,
    timeout_seconds: int,
) -> Dict:
    """
    Execute a single test iteration.
    
    Args:
        cycle_count: Number of cycles to run
        test_name: Display name for test
        timeout_seconds: Max allowed time
    
    Returns:
        dict with test results
    """
    print(f"\n[TEST-MATRIX] Starting {test_name}...")
    start_time = time.time()
    
    output_dir = PROJECT_ROOT / "h2q_project" / "reports" / f"test_matrix_phase1_{test_name.lower()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable,
        "-m",
        "h2q_project.tools.start_quantum_agi_highdim_evolution",
        "--cycles", str(cycle_count),
        "--output", str(output_dir),
        "--resource-profile", "low",
        "--projection-dim", "64",
        "--parallel-branches", "3",
        "--important-cycle-every", "2",
        "--force-acceptance-prompt",
        "--strict-acceptance",
    ]
    
    try:
        # Run the test
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_seconds
        )
        
        elapsed = time.time() - start_time
        
        # Extract results from acceptance JSON
        acceptance_json = output_dir / "quantum_agi_highdim_acceptance.json"
        acceptance_data = {}
        uplift_value = None
        acceptance_passed = False
        
        if acceptance_json.exists():
            try:
                with open(acceptance_json) as f:
                    acceptance_data = json.load(f)
                    acceptance_passed = acceptance_data.get("passed", False)
                    # Find uplift value in criteria
                    for criterion in acceptance_data.get("criteria", []):
                        if criterion.get("name") == "composite_uplift":
                            uplift_value = criterion.get("value")
            except Exception as e:
                print(f"  [ERROR] Failed to parse acceptance.json: {e}")
        
        status = "✅ PASS" if acceptance_passed else "❌ FAIL"
        
        test_result = {
            "test_name": test_name,
            "cycle_count": cycle_count,
            "status": "success",
            "acceptance_passed": acceptance_passed,
            "uplift_value": uplift_value,
            "elapsed_seconds": elapsed,
            "output_dir": str(output_dir),
        }
        
        print(f"  {status} | cycles={cycle_count} | uplift={uplift_value:.4f if uplift_value else 'N/A'} | time={elapsed:.1f}s")
        
        return test_result
        
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"  ❌ TIMEOUT | Exceeded {timeout_seconds}s (elapsed={elapsed:.1f}s)")
        return {
            "test_name": test_name,
            "cycle_count": cycle_count,
            "status": "timeout",
            "elapsed_seconds": elapsed,
        }
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  ❌ ERROR | {e}")
        return {
            "test_name": test_name,
            "cycle_count": cycle_count,
            "status": "error",
            "error": str(e),
            "elapsed_seconds": elapsed,
        }


def main():
    """Run the full test matrix."""
    print("=" * 72)
    print("Phase 1 Track A - Uplift Stability Test Matrix")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 72)
    
    # Test matrix configuration
    # (cycle_count, test_name, timeout_seconds, uplift_target)
    test_matrix = [
        (12, "SMOKE_12CYCLE", 900, -0.010),      # 15min timeout, target -0.010
        (24, "BALANCED_24CYCLE", 2700, -0.005),  # 45min timeout, target -0.005
        (48, "ENDURANCE_48CYCLE", 10800, 0.000), # 3hr timeout, target ≥0.000
    ]
    
    results: List[Dict] = []
    
    for cycle_count, test_name, timeout_sec, target_uplift in test_matrix:
        result = run_test_iteration(cycle_count, test_name, timeout_sec)
        result["target_uplift"] = target_uplift
        results.append(result)
    
    # Summary report
    print("\n" + "=" * 72)
    print("TEST MATRIX SUMMARY")
    print("=" * 72)
    
    comparison_table = []
    for i, (result, (_, test_name, _, target)) in enumerate(zip(results, test_matrix)):
        if result["status"] == "success":
            uplift = result.get("uplift_value")
            passed = result.get("acceptance_passed", False)
            meets_target = (uplift is not None) and (uplift >= target)
            status_icon = "✅" if meets_target else "⚠️"
            
            comparison_table.append({
                "test": test_name,
                "cycles": result["cycle_count"],
                "uplift": f"{uplift:.6f}" if uplift is not None else "N/A",
                "target": f"{target:.6f}",
                "meets_target": meets_target,
                "acceptance": passed,
                "time_seconds": f"{result['elapsed_seconds']:.1f}",
                "status": status_icon,
            })
        else:
            comparison_table.append({
                "test": test_name,
                "cycles": result["cycle_count"],
                "uplift": "FAILED",
                "target": f"{target:.6f}",
                "meets_target": False,
                "acceptance": False,
                "time_seconds": f"{result['elapsed_seconds']:.1f}",
                "status": "❌",
            })
    
    # Print table
    print("\n| Status | Test Name | Cycles | Uplift | Target | Meets? | Accept | Time ")
    print("|--------|-----------|--------|--------|--------|--------|--------|------")
    for row in comparison_table:
        print(
            f"| {row['status']} | {row['test']:20s} | {row['cycles']:6d} | {row['uplift']:>8s} | "
            f"{row['target']:>8s} | {'✅' if row['meets_target'] else '❌':^6s} | "
            f"{'✅' if row['acceptance'] else '❌':^6s} | {row['time_seconds']:>7s}s"
        )
    
    # Overall summary
    all_passed = all(r.get("meets_target", False) for r in comparison_table)
    print("\n" + "=" * 72)
    if all_passed:
        print("🎉 ALL TESTS PASSED - Phase 1 Track A Uplift Stability Verified")
    else:
        print("⚠️  SOME TESTS FAILED - Review uplift recovery strategy")
    print("=" * 72)
    
    # Save detailed results
    results_path = PROJECT_ROOT / "h2q_project" / "reports" / "test_matrix_phase1_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": {
                "resource_profile": "low",
                "projection_dim": 64,
                "parallel_branches": 3,
                "important_cycle_every": 2,
                "force_acceptance_prompt": True,
                "strict_acceptance": True,
            },
            "comparison": comparison_table,
            "all_passed": all_passed,
        }, f, indent=2)
    print(f"\nDetailed results saved to: {results_path}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
