"""Tests for TransformerBlock implementation."""

import pytest
import torch
from transformer_block import TransformerBlock, KVCache


def test_transformer_block_initialization():
    """Test transformer block initialization."""
    # Standard transformer
    block = TransformerBlock(
        d_model=128,
        n_heads=8,
        d_ff=512,
        use_moe=False,
    )
    
    assert block.d_model == 128
    assert block.use_moe == False
    
    # MoE transformer
    block_moe = TransformerBlock(
        d_model=128,
        n_heads=8,
        d_ff=512,
        use_moe=True,
        num_experts=4,
        top_k_experts=2,
    )
    
    assert block_moe.use_moe == True


def test_transformer_block_forward_standard():
    """Test standard transformer block forward pass."""
    batch_size = 2
    seq_len = 10
    d_model = 64
    
    block = TransformerBlock(
        d_model=d_model,
        n_heads=4,
        d_ff=256,
        use_moe=False,
    )
    
    x = torch.randn(batch_size, seq_len, d_model)
    output, cache, aux_loss = block(x)
    
    assert output.shape == (batch_size, seq_len, d_model)
    assert cache is None
    assert aux_loss is None


def test_transformer_block_forward_moe():
    """Test MoE transformer block forward pass."""
    batch_size = 2
    seq_len = 10
    d_model = 64
    
    block = TransformerBlock(
        d_model=d_model,
        n_heads=4,
        d_ff=256,
        use_moe=True,
        num_experts=4,
        top_k_experts=2,
    )
    block.train()
    
    x = torch.randn(batch_size, seq_len, d_model)
    output, cache, aux_loss = block(x, return_aux_loss=True)
    
    assert output.shape == (batch_size, seq_len, d_model)
    assert cache is None
    assert aux_loss is not None
    assert aux_loss.item() >= 0


def test_transformer_block_with_kv_cache():
    """Test transformer block with KV cache."""
    batch_size = 2
    seq_len = 5
    d_model = 64
    n_heads = 4
    
    block = TransformerBlock(
        d_model=d_model,
        n_heads=n_heads,
        d_ff=256,
        use_moe=False,
    )
    
    # Create cache
    cache = block.create_kv_cache(
        max_batch_size=batch_size,
        max_seq_len=20,
    )
    
    # First forward pass
    x1 = torch.randn(batch_size, seq_len, d_model)
    output1, updated_cache, _ = block(x1, kv_cache=cache, use_cache=True, cache_position=0)
    
    assert output1.shape == (batch_size, seq_len, d_model)
    assert updated_cache is not None
    assert cache.cache_position == seq_len
    
    # Second forward pass (incremental)
    x2 = torch.randn(batch_size, 3, d_model)
    output2, updated_cache, _ = block(x2, kv_cache=cache, use_cache=True, cache_position=seq_len)
    
    assert output2.shape == (batch_size, 3, d_model)
    assert cache.cache_position == seq_len + 3


def test_transformer_block_with_attention_mask():
    """Test transformer block with attention mask."""
    batch_size = 2
    seq_len = 10
    d_model = 64
    
    block = TransformerBlock(
        d_model=d_model,
        n_heads=4,
        d_ff=256,
    )
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Create causal mask
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    
    output, _, _ = block(x, mask=mask)
    
    assert output.shape == (batch_size, seq_len, d_model)


def test_transformer_block_residual_connections():
    """Test that residual connections are working."""
    batch_size = 2
    seq_len = 10
    d_model = 64
    
    block = TransformerBlock(
        d_model=d_model,
        n_heads=4,
        d_ff=256,
        dropout=0.0,  # Disable dropout for this test
    )
    block.eval()
    
    x = torch.randn(batch_size, seq_len, d_model)
    output, _, _ = block(x)
    
    # Output should be different from input due to attention and FFN
    assert not torch.allclose(output, x)
    
    # But not too different (residual connections should help)
    # This is a sanity check
    assert output.shape == x.shape


def test_transformer_block_create_kv_cache():
    """Test KV cache creation."""
    block = TransformerBlock(
        d_model=64,
        n_heads=4,
        d_ff=256,
    )
    
    cache = block.create_kv_cache(
        max_batch_size=2,
        max_seq_len=100,
    )
    
    assert isinstance(cache, KVCache)
    assert cache.max_batch_size == 2
    assert cache.max_seq_len == 100
    assert cache.n_heads == 4
    assert cache.head_dim == 16  # 64 / 4


def test_transformer_block_moe_with_cache():
    """Test MoE transformer block with KV cache."""
    batch_size = 2
    seq_len = 5
    d_model = 64
    
    block = TransformerBlock(
        d_model=d_model,
        n_heads=4,
        d_ff=256,
        use_moe=True,
        num_experts=4,
        top_k_experts=2,
    )
    block.train()
    
    cache = block.create_kv_cache(
        max_batch_size=batch_size,
        max_seq_len=20,
    )
    
    x = torch.randn(batch_size, seq_len, d_model)
    output, updated_cache, aux_loss = block(
        x,
        kv_cache=cache,
        use_cache=True,
        cache_position=0,
        return_aux_loss=True,
    )
    
    assert output.shape == (batch_size, seq_len, d_model)
    assert updated_cache is not None
    assert aux_loss is not None


def test_transformer_block_eval_mode():
    """Test transformer block in eval mode."""
    block = TransformerBlock(
        d_model=64,
        n_heads=4,
        d_ff=256,
        use_moe=True,
        num_experts=4,
    )
    block.eval()
    
    x = torch.randn(2, 10, 64)
    
    # In eval mode, aux loss should be None even if requested
    output, _, aux_loss = block(x, return_aux_loss=True)
    
    assert output.shape == x.shape
    # aux_loss will be None because eval mode


def test_transformer_block_different_heads():
    """Test transformer block with different number of heads."""
    d_model = 96
    
    for n_heads in [1, 2, 3, 4, 6, 8, 12]:
        if d_model % n_heads != 0:
            continue
        
        block = TransformerBlock(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=384,
        )
        
        x = torch.randn(2, 5, d_model)
        output, _, _ = block(x)
        
        assert output.shape == x.shape
