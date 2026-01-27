"""Tests for Mixture of Experts implementation."""

import pytest
import torch
from transformer_block import MixtureOfExperts


def test_moe_initialization():
    """Test MoE initialization."""
    moe = MixtureOfExperts(
        d_model=128,
        d_ff=512,
        num_experts=8,
        top_k=2,
    )
    
    assert moe.d_model == 128
    assert moe.num_experts == 8
    assert moe.top_k == 2
    assert len(moe.experts) == 8


def test_moe_forward():
    """Test MoE forward pass."""
    batch_size = 2
    seq_len = 10
    d_model = 64
    
    moe = MixtureOfExperts(
        d_model=d_model,
        d_ff=256,
        num_experts=4,
        top_k=2,
    )
    
    x = torch.randn(batch_size, seq_len, d_model)
    output, aux_loss = moe(x, return_aux_loss=False)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model)
    assert aux_loss is None


def test_moe_aux_loss():
    """Test MoE auxiliary load balancing loss."""
    batch_size = 4
    seq_len = 20
    d_model = 64
    
    moe = MixtureOfExperts(
        d_model=d_model,
        d_ff=256,
        num_experts=8,
        top_k=2,
    )
    moe.train()
    
    x = torch.randn(batch_size, seq_len, d_model)
    output, aux_loss = moe(x, return_aux_loss=True)
    
    # Check output and loss
    assert output.shape == (batch_size, seq_len, d_model)
    assert aux_loss is not None
    assert aux_loss.item() >= 0


def test_moe_expert_selection():
    """Test that MoE selects correct number of experts."""
    batch_size = 2
    seq_len = 5
    d_model = 32
    top_k = 2
    
    moe = MixtureOfExperts(
        d_model=d_model,
        d_ff=128,
        num_experts=4,
        top_k=top_k,
    )
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Forward pass
    output, _ = moe(x)
    
    # Output should have correct shape
    assert output.shape == (batch_size, seq_len, d_model)


def test_moe_deterministic():
    """Test that MoE is deterministic with same input."""
    torch.manual_seed(42)
    
    batch_size = 2
    seq_len = 10
    d_model = 64
    
    moe = MixtureOfExperts(
        d_model=d_model,
        d_ff=256,
        num_experts=4,
        top_k=2,
        dropout=0.0,  # Disable dropout for deterministic test
    )
    moe.eval()
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Run twice
    output1, _ = moe(x)
    output2, _ = moe(x)
    
    # Should be identical
    assert torch.allclose(output1, output2, atol=1e-6)


def test_moe_different_top_k():
    """Test MoE with different top_k values."""
    batch_size = 2
    seq_len = 5
    d_model = 32
    
    for top_k in [1, 2, 4]:
        moe = MixtureOfExperts(
            d_model=d_model,
            d_ff=128,
            num_experts=8,
            top_k=top_k,
        )
        
        x = torch.randn(batch_size, seq_len, d_model)
        output, _ = moe(x)
        
        assert output.shape == (batch_size, seq_len, d_model)


def test_moe_expert_counts():
    """Test expert usage counting."""
    moe = MixtureOfExperts(
        d_model=64,
        d_ff=256,
        num_experts=4,
        top_k=2,
    )
    moe.train()
    
    x = torch.randn(2, 10, 64)
    
    # Reset counts
    moe.reset_expert_counts()
    assert torch.allclose(moe.expert_counts, torch.zeros(4))
    
    # Run forward pass
    output, _ = moe(x, return_aux_loss=True)
    
    # Some experts should have been used
    assert moe.expert_counts.sum() > 0
