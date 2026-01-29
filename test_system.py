#!/usr/bin/env python3
import torch
from agi_evolution_loss_metrics import create_agi_evolution_loss_system, MathematicalCoreMetrics

system = create_agi_evolution_loss_system()
capability_embeddings = {'mathematical_reasoning': torch.randn(256), 'creative_problem_solving': torch.randn(256), 'knowledge_integration': torch.randn(256), 'emergent_capabilities': torch.randn(256)}
current_performance = {'mathematical_reasoning': 0.8, 'creative_problem_solving': 0.7, 'knowledge_integration': 0.6, 'emergent_capabilities': 0.5}
new_knowledge = torch.randn(256)
existing_knowledge = [torch.randn(256) for _ in range(3)]
current_state = torch.randn(256)
mathematical_metrics = MathematicalCoreMetrics()

try:
    result = system(capability_embeddings, current_performance, new_knowledge, existing_knowledge, current_state, mathematical_metrics)
    print('AGI_EvolutionLossSystem success:', result.total_loss)
except Exception as e:
    print('AGI_EvolutionLossSystem error:', e)
    import traceback
    traceback.print_exc()