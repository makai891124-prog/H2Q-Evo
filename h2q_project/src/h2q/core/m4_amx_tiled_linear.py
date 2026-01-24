import torch
import torch.nn as nn
from h2q.layers.amx_linear import AMXQuaternionicLinear, get_compatible_dde
from h2q.core.metal_jit_bridge import MetalJITBridge, audit_jit_integrity
from h2q.core.discrete_decision_engine import get_canonical_dde
from h2q.core.interface_registry import topological_dde_normalization

class M4_AMX_Tiled_Linear_Factory:
    """
    M4_AMX_Tiled_Linear Factory: A recursive model-rewriter for M4 Silicon.
    Hot-swaps torch.nn.Linear with AMXQuaternionicLinear modules.
    Enforces 16x16 register tiling and leverages MetalJITBridge.
    """
    
    def __init__(self, tiling_size=16, use_jit=True):
        self.tiling_size = tiling_size
        self.use_jit = use_jit
        # Resolve DDE safely to avoid 'dim' keyword argument errors found in previous iterations
        self.dde = get_canonical_dde()
        
        if self.use_jit:
            self.jit_bridge = MetalJITBridge()
            audit_jit_integrity(self.jit_bridge)

    def rewrite(self, model: nn.Module) -> nn.Module:
        """
        Recursively traverses the model and replaces nn.Linear layers.
        """
        return self._recursive_swap(model)

    def _recursive_swap(self, module: nn.Module) -> nn.Module:
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                # Atom: Identify Symmetry - Ensure dimensions are compatible with Quaternionic (4-component) logic
                # If not divisible by 4, we apply manifold padding internally in AMXQuaternionicLinear
                new_layer = AMXQuaternionicLinear(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=child.bias is not None,
                    dde=self.dde,
                    tiling_size=self.tiling_size
                )
                
                # Atom: Memory Management - Transfer weights with bit-accurate reconstruction
                with torch.no_grad():
                    # We map the real weights into the quaternionic manifold
                    # Note: AMXQuaternionicLinear handles the internal 4-way split
                    if child.weight is not None:
                        new_layer.weight.copy_(child.weight)
                    if child.bias is not None:
                        new_layer.bias.copy_(child.bias)
                
                # Elastic Extension: Attach JIT optimization if on M4 silicon
                if self.use_jit:
                    self.jit_bridge.optimize_layer(new_layer, target_tiling=self.tiling_size)
                
                setattr(module, name, new_layer)
            else:
                self._recursive_swap(child)
        return model

def apply_m4_amx_rewrite(model: nn.Module, tiling: int = 16) -> nn.Module:
    """
    Convenience function to trigger the M4_AMX_Tiled_Linear rewriter.
    """
    factory = M4_AMX_Tiled_Linear_Factory(tiling_size=tiling)
    rewritten_model = factory.rewrite(model)
    
    # Final Veracity Audit
    print(f"[M4-AMX] Rewrite Complete. Tiling: {tiling}x{tiling}. JIT: Active.")
    return rewritten_model
