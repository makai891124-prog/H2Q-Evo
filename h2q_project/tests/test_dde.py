# tests/test_dde.py

import torch
import pytest
from h2q.core.discrete_decision_engine import DiscreteDecisionEngine, get_canonical_dde, LatentConfig

def test_dde_basic_functionality():
    """Test basic DDE functionality with new API."""
    context_dim = 16
    action_dim = 4

    # Create DDE with new API
    dde = get_canonical_dde(latent_dim=context_dim, n_choices=action_dim)

    # Test forward pass
    context = torch.randn(1, context_dim)
    logits = dde(context)

    # Check output shape
    assert logits.shape == (1, action_dim)

    # Test with eta modulation
    eta = torch.tensor([[0.1]])
    logits_with_eta = dde(context, eta=eta)
    assert logits_with_eta.shape == (1, action_dim)

    # Verify eta affects the output
    assert not torch.allclose(logits, logits_with_eta)

def test_dde_config_handling():
    """Test DDE configuration handling."""
    # Test with dict config
    config_dict = {
        "latent_dim": 32,
        "n_choices": 8,
        "temperature": 0.5
    }
    dde = get_canonical_dde(config=config_dict)
    assert dde.config.latent_dim == 32
    assert dde.config.n_choices == 8
    assert dde.config.temperature == 0.5

    # Test with LatentConfig object
    config_obj = LatentConfig(latent_dim=64, n_choices=16)
    dde2 = DiscreteDecisionEngine(config=config_obj)
    assert dde2.config.latent_dim == 64
    assert dde2.config.n_choices == 16

def test_dde_backward_compatibility():
    """Test backward compatibility with legacy parameters."""
    # Test legacy parameter mapping
    dde = DiscreteDecisionEngine(dim=128, action_dim=32)
    assert dde.config.latent_dim == 128
    assert dde.config.n_choices == 32

    # Test context_dim mapping
    dde2 = DiscreteDecisionEngine(context_dim=64, action_dim=16)
    assert dde2.config.latent_dim == 64
    assert dde2.config.n_choices == 16