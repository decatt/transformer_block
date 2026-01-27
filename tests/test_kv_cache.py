"""Tests for KV Cache implementation."""

import pytest
import torch
from transformer_block import KVCache


def test_kv_cache_initialization():
    """Test KV cache initialization."""
    cache = KVCache(
        max_batch_size=2,
        max_seq_len=10,
        n_heads=4,
        head_dim=8,
    )
    
    assert cache.max_batch_size == 2
    assert cache.max_seq_len == 10
    assert cache.n_heads == 4
    assert cache.head_dim == 8
    assert cache.cache_position == 0
    assert cache.k_cache.shape == (2, 4, 10, 8)
    assert cache.v_cache.shape == (2, 4, 10, 8)


def test_kv_cache_update():
    """Test updating KV cache."""
    cache = KVCache(
        max_batch_size=2,
        max_seq_len=10,
        n_heads=4,
        head_dim=8,
    )
    
    # Create new keys and values
    k = torch.randn(2, 4, 3, 8)  # 3 new tokens
    v = torch.randn(2, 4, 3, 8)
    
    # Update cache
    cached_k, cached_v = cache.update(k, v)
    
    # Check shapes
    assert cached_k.shape == (2, 4, 3, 8)
    assert cached_v.shape == (2, 4, 3, 8)
    assert cache.cache_position == 3
    
    # Check values match
    assert torch.allclose(cached_k, k)
    assert torch.allclose(cached_v, v)


def test_kv_cache_incremental_update():
    """Test incremental updates to KV cache."""
    cache = KVCache(
        max_batch_size=1,
        max_seq_len=10,
        n_heads=2,
        head_dim=4,
    )
    
    # First update
    k1 = torch.ones(1, 2, 2, 4)
    v1 = torch.ones(1, 2, 2, 4) * 2
    cached_k1, cached_v1 = cache.update(k1, v1)
    
    assert cached_k1.shape == (1, 2, 2, 4)
    assert cache.cache_position == 2
    
    # Second update
    k2 = torch.ones(1, 2, 3, 4) * 3
    v2 = torch.ones(1, 2, 3, 4) * 4
    cached_k2, cached_v2 = cache.update(k2, v2)
    
    assert cached_k2.shape == (1, 2, 5, 4)
    assert cache.cache_position == 5
    
    # Check that old values are preserved
    assert torch.allclose(cached_k2[:, :, :2, :], k1)
    assert torch.allclose(cached_v2[:, :, :2, :], v1)
    assert torch.allclose(cached_k2[:, :, 2:, :], k2)
    assert torch.allclose(cached_v2[:, :, 2:, :], v2)


def test_kv_cache_get():
    """Test getting values from cache."""
    cache = KVCache(
        max_batch_size=2,
        max_seq_len=10,
        n_heads=4,
        head_dim=8,
    )
    
    k = torch.randn(2, 4, 5, 8)
    v = torch.randn(2, 4, 5, 8)
    cache.update(k, v)
    
    # Get full cache
    cached_k, cached_v = cache.get()
    assert cached_k.shape == (2, 4, 5, 8)
    assert cached_v.shape == (2, 4, 5, 8)
    
    # Get partial cache
    cached_k, cached_v = cache.get(batch_size=1, seq_len=3)
    assert cached_k.shape == (1, 4, 3, 8)
    assert cached_v.shape == (1, 4, 3, 8)


def test_kv_cache_reset():
    """Test resetting cache."""
    cache = KVCache(
        max_batch_size=2,
        max_seq_len=10,
        n_heads=4,
        head_dim=8,
    )
    
    k = torch.randn(2, 4, 5, 8)
    v = torch.randn(2, 4, 5, 8)
    cache.update(k, v)
    
    assert cache.cache_position == 5
    
    cache.reset()
    
    assert cache.cache_position == 0
    assert torch.allclose(cache.k_cache, torch.zeros_like(cache.k_cache))
    assert torch.allclose(cache.v_cache, torch.zeros_like(cache.v_cache))


def test_kv_cache_device_transfer():
    """Test moving cache to different device."""
    cache = KVCache(
        max_batch_size=2,
        max_seq_len=10,
        n_heads=4,
        head_dim=8,
        device="cpu",
    )
    
    assert cache.device == "cpu"
    assert cache.k_cache.device.type == "cpu"
    
    # Note: This test will only work on systems with CUDA
    # For CI/CD without GPU, it will just test CPU to CPU
    cache.to("cpu")
    assert cache.device == "cpu"
