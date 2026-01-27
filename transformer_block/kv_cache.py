"""KV Cache implementation for efficient transformer inference."""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class KVCache:
    """Key-Value cache for storing and reusing computed keys and values in attention.
    
    This cache enables efficient autoregressive generation by avoiding recomputation
    of keys and values for previously processed tokens.
    
    Args:
        max_batch_size: Maximum batch size supported
        max_seq_len: Maximum sequence length supported
        n_heads: Number of attention heads
        head_dim: Dimension of each attention head
        dtype: Data type for cache tensors
        device: Device to store cache tensors
    """
    
    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        n_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        
        # Initialize cache tensors
        self.k_cache = torch.zeros(
            (max_batch_size, n_heads, max_seq_len, head_dim),
            dtype=dtype,
            device=device,
        )
        self.v_cache = torch.zeros(
            (max_batch_size, n_heads, max_seq_len, head_dim),
            dtype=dtype,
            device=device,
        )
        
        # Track current position in cache
        self.cache_position = 0
    
    def update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        start_pos: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache with new keys and values.
        
        Args:
            k: New keys tensor of shape (batch_size, n_heads, seq_len, head_dim)
            v: New values tensor of shape (batch_size, n_heads, seq_len, head_dim)
            start_pos: Starting position in cache. If None, uses current cache_position
        
        Returns:
            Tuple of (cached_keys, cached_values) including both old and new values
        """
        batch_size, n_heads, seq_len, head_dim = k.shape
        
        if start_pos is None:
            start_pos = self.cache_position
        
        # Update cache
        end_pos = start_pos + seq_len
        self.k_cache[:batch_size, :, start_pos:end_pos, :] = k
        self.v_cache[:batch_size, :, start_pos:end_pos, :] = v
        
        # Update position
        self.cache_position = end_pos
        
        # Return full cache up to current position
        return (
            self.k_cache[:batch_size, :, :end_pos, :],
            self.v_cache[:batch_size, :, :end_pos, :],
        )
    
    def get(
        self,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get current cached keys and values.
        
        Args:
            batch_size: Batch size to return. If None, returns full batch dimension
            seq_len: Sequence length to return. If None, returns up to cache_position
        
        Returns:
            Tuple of (cached_keys, cached_values)
        """
        if batch_size is None:
            batch_size = self.max_batch_size
        if seq_len is None:
            seq_len = self.cache_position
        
        return (
            self.k_cache[:batch_size, :, :seq_len, :],
            self.v_cache[:batch_size, :, :seq_len, :],
        )
    
    def reset(self):
        """Reset cache to initial state."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.cache_position = 0
    
    def to(self, device: str):
        """Move cache to specified device."""
        self.device = device
        self.k_cache = self.k_cache.to(device)
        self.v_cache = self.v_cache.to(device)
        return self
