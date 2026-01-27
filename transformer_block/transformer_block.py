"""Transformer block with KV cache and Mixture of Experts support."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .kv_cache import KVCache
from .moe import MixtureOfExperts


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional KV cache support.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout probability
        use_bias: Whether to use bias in linear layers
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        use_bias: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=use_bias)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = False,
        cache_position: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        """Forward pass with optional KV caching.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Attention mask of shape (batch_size, seq_len, seq_len) or broadcastable
            kv_cache: Optional KV cache for incremental decoding
            use_cache: Whether to use/update the cache
            cache_position: Position in cache to start from
        
        Returns:
            Tuple of (output, updated_kv_cache)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention: (batch_size, n_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Use KV cache if provided
        if use_cache and kv_cache is not None:
            k, v = kv_cache.update(k, v, start_pos=cache_position)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(attn_output)
        
        return output, kv_cache if use_cache else None


class FeedForward(nn.Module):
    """Standard feed-forward network.
    
    Args:
        d_model: Model dimension
        d_ff: Hidden dimension
        dropout: Dropout probability
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with support for KV cache and Mixture of Experts.
    
    This block can operate in three modes:
    1. Standard transformer (use_moe=False)
    2. MoE transformer (use_moe=True)
    3. With KV cache for efficient inference (use_cache=True)
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        dropout: Dropout probability
        use_moe: Whether to use Mixture of Experts for feed-forward
        num_experts: Number of experts (only used if use_moe=True)
        top_k_experts: Number of experts to activate (only used if use_moe=True)
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_moe: bool = False,
        num_experts: int = 8,
        top_k_experts: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_moe = use_moe
        
        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Feed-forward or MoE
        if use_moe:
            self.feed_forward = MixtureOfExperts(
                d_model=d_model,
                d_ff=d_ff,
                num_experts=num_experts,
                top_k=top_k_experts,
                dropout=dropout,
            )
        else:
            self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = False,
        cache_position: Optional[int] = None,
        return_aux_loss: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache], Optional[torch.Tensor]]:
        """Forward pass through transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            kv_cache: Optional KV cache for attention
            use_cache: Whether to use/update KV cache
            cache_position: Position in cache
            return_aux_loss: Whether to return auxiliary MoE load balancing loss
        
        Returns:
            Tuple of (output, updated_kv_cache, aux_loss)
        """
        # Self-attention with residual connection
        attn_output, updated_cache = self.attention(
            self.norm1(x),
            mask=mask,
            kv_cache=kv_cache,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual connection
        ff_input = self.norm2(x)
        
        aux_loss = None
        if self.use_moe:
            ff_output, aux_loss = self.feed_forward(ff_input, return_aux_loss=return_aux_loss)
        else:
            ff_output = self.feed_forward(ff_input)
        
        x = x + self.dropout(ff_output)
        
        return x, updated_cache, aux_loss
    
    def create_kv_cache(
        self,
        max_batch_size: int,
        max_seq_len: int,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> KVCache:
        """Create a KV cache for this transformer block.
        
        Args:
            max_batch_size: Maximum batch size
            max_seq_len: Maximum sequence length
            dtype: Data type for cache
            device: Device for cache
        
        Returns:
            Initialized KVCache instance
        """
        n_heads = self.attention.n_heads
        head_dim = self.attention.head_dim
        
        return KVCache(
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            n_heads=n_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )
