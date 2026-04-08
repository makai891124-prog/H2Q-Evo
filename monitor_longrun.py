#!/usr/bin/env python3
"""Monitor the formal longrun progress."""
import json
import time
import sys
from pathlib import Path

cycles_file = Path("outputs/formal_longrun_12cycles_2026/quantum_agi_highdim_cycles.jsonl")

print("\n" + "="*80)
print("MONITORING: formal_longrun_12cycles_2026 execution")
print("="*80)

last_cycle_count = 0
check_count = 0
max_checks = 150  # ~12-13 minutes

while check_count < max_checks:
    check_count += 1
    elapsed_min = check_count * 5 / 60
    
    if cycles_file.exists():
        try:
            with open(cycles_file) as f:
                lines = f.readlines()
            
            current_count = len(lines)
            if current_count > last_cycle_count:
                print(f"\n[{elapsed_min:.1f} min] {current_count} cycles completed")
                
                # Parse and show last few cycles
                visible_cycles = []
                for i, cycle_data in enumerate(lines[-min(5, current_count):]):
                    try:
                        data = json.loads(cycle_data)
                        c = data.get('cycle', 0)
                        comp = data.get('composite_score', 0.0)
                        cap = data.get('capability', {}).get('overall_score', 0.0)
                        cons = data.get('highdim', {}).get('consensus_score', 0.0)
                        ent = data.get('quantum', {}).get('entanglement_negative_ratio', 0.0)
                        visible_cycles.append({
                            'c': c,
                            'comp': comp,
                            'cap': cap,
                            'cons': cons,
                            'ent': ent
                        })
                    except:
                        pass
                
                for item in visible_cycles[-3:]:
                    print(f"    Cycle {item['c']:3d}: composite={item['comp']:.4f}, "
                          f"capability={item['cap']:.1f}%, consensus={item['cons']:.4f}, "
                          f"entanglement={item['ent']:.3f}")
                
                last_cycle_count = current_count
                
                # Check for target or early completion
                if current_count >= 12:
                    print("\n[SUCCESS] Target of 12+ cycles reached!")
                    # Show final summary
                    all_cycles = [json.loads(line) for line in lines]
                    composite_scores = [c.get('composite_score', 0.0) for c in all_cycles]
                    initial = composite_scores[:max(1, len(composite_scores)//4)]
                    final = composite_scores[-max(1, len(composite_scores)//4):]
                    uplift = sum(final)/len(final) - sum(initial)/len(initial)
                    print(f"\nComposite Uplift: {uplift:.4f}")
                    print(f"Initial mean: {sum(initial)/len(initial):.4f}")
                    print(f"Final mean: {sum(final)/len(final):.4f}")
                    sys.exit(0)
        except Exception as e:
            pass  # File may be locked, just wait
    
    if check_count % 6 == 0:  # Every 30 seconds
        print(f"[{elapsed_min:.1f} min] Waiting for more cycles...")
    
    time.sleep(5)

print("\n[TIMEOUT] Monitoring limit reached without completing 12 cycles")
sys.exit(1)
