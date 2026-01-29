#!/usr/bin/env python3
import torch
from agi_evolution_loss_integration import get_evolution_loss_integrator

print("Starting debug...")
integrator = get_evolution_loss_integrator()
print("Integrator created")

try:
    result = integrator.compute_evolution_loss(
        {'mathematical_reasoning': torch.randn(256), 'creative_problem_solving': torch.randn(256), 'knowledge_integration': torch.randn(256), 'emergent_capabilities': torch.randn(256)},
        {'mathematical_reasoning': 0.8, 'creative_problem_solving': 0.7, 'knowledge_integration': 0.6, 'emergent_capabilities': 0.5},
        torch.randn(256), [torch.randn(256)], torch.randn(256),
        {'statistics': {'avg_constraint_violation': 0.1, 'avg_fueter_violation': 0.05}, 'forward_count': 100, 'enabled_modules': {'lie_automorphism': True, 'reflection_operators': True, 'knot_constraints': True, 'dde_integration': True}}
    )
    print("Result computed successfully")
    print("Result:", result is not None)
except Exception as e:
    print("Error in compute_evolution_loss:", e)
    import traceback
    traceback.print_exc()