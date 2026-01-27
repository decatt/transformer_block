"""Example usage of transformer_block with KV cache and MoE."""

import torch
from transformer_block import TransformerBlock


def example_standard_transformer():
    """Example: Standard transformer block without MoE."""
    print("=" * 60)
    print("Example 1: Standard Transformer Block")
    print("=" * 60)
    
    # Create a standard transformer block
    block = TransformerBlock(
        d_model=128,
        n_heads=8,
        d_ff=512,
        dropout=0.1,
        use_moe=False,
    )
    
    # Input: (batch_size=2, seq_len=10, d_model=128)
    x = torch.randn(2, 10, 128)
    
    # Forward pass
    output, _, _ = block(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print()


def example_moe_transformer():
    """Example: Transformer block with Mixture of Experts."""
    print("=" * 60)
    print("Example 2: Transformer Block with MoE")
    print("=" * 60)
    
    # Create a transformer block with MoE
    block = TransformerBlock(
        d_model=128,
        n_heads=8,
        d_ff=512,
        dropout=0.1,
        use_moe=True,
        num_experts=8,
        top_k_experts=2,
    )
    block.train()
    
    # Input
    x = torch.randn(2, 10, 128)
    
    # Forward pass with auxiliary loss
    output, _, aux_loss = block(x, return_aux_loss=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Auxiliary loss: {aux_loss.item():.4f}")
    print()


def example_kv_cache():
    """Example: Using KV cache for efficient inference."""
    print("=" * 60)
    print("Example 3: Transformer Block with KV Cache")
    print("=" * 60)
    
    # Create transformer block
    block = TransformerBlock(
        d_model=64,
        n_heads=4,
        d_ff=256,
        dropout=0.0,
    )
    block.eval()
    
    # Create KV cache
    cache = block.create_kv_cache(
        max_batch_size=1,
        max_seq_len=100,
    )
    
    # First step: process initial sequence
    x1 = torch.randn(1, 5, 64)
    output1, _, _ = block(x1, kv_cache=cache, use_cache=True, cache_position=0)
    
    print(f"Step 1:")
    print(f"  Input shape: {x1.shape}")
    print(f"  Output shape: {output1.shape}")
    print(f"  Cache position: {cache.cache_position}")
    
    # Second step: process next token (autoregressive generation)
    x2 = torch.randn(1, 1, 64)
    output2, _, _ = block(x2, kv_cache=cache, use_cache=True, cache_position=5)
    
    print(f"Step 2:")
    print(f"  Input shape: {x2.shape}")
    print(f"  Output shape: {output2.shape}")
    print(f"  Cache position: {cache.cache_position}")
    
    # Third step: process another token
    x3 = torch.randn(1, 1, 64)
    output3, _, _ = block(x3, kv_cache=cache, use_cache=True, cache_position=6)
    
    print(f"Step 3:")
    print(f"  Input shape: {x3.shape}")
    print(f"  Output shape: {output3.shape}")
    print(f"  Cache position: {cache.cache_position}")
    print()


def example_moe_with_cache():
    """Example: Combining MoE and KV cache."""
    print("=" * 60)
    print("Example 4: MoE Transformer with KV Cache")
    print("=" * 60)
    
    # Create MoE transformer block
    block = TransformerBlock(
        d_model=64,
        n_heads=4,
        d_ff=256,
        dropout=0.0,
        use_moe=True,
        num_experts=4,
        top_k_experts=2,
    )
    block.eval()
    
    # Create KV cache
    cache = block.create_kv_cache(
        max_batch_size=2,
        max_seq_len=50,
    )
    
    # Process sequence with cache
    x = torch.randn(2, 10, 64)
    output, _, _ = block(x, kv_cache=cache, use_cache=True, cache_position=0)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Cache position: {cache.cache_position}")
    print(f"Using MoE: {block.use_moe}")
    print()


def example_with_attention_mask():
    """Example: Using attention mask for causal attention."""
    print("=" * 60)
    print("Example 5: Transformer with Causal Attention Mask")
    print("=" * 60)
    
    # Create transformer block
    block = TransformerBlock(
        d_model=64,
        n_heads=4,
        d_ff=256,
    )
    block.eval()
    
    # Input
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, 64)
    
    # Create causal mask (lower triangular)
    # Shape: (1, 1, seq_len, seq_len) for broadcasting
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    
    # Forward pass with mask
    output, _, _ = block(x, mask=mask)
    
    print(f"Input shape: {x.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Output shape: {output.shape}")
    print("Causal mask ensures each position can only attend to previous positions")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Transformer Block Examples")
    print("=" * 60 + "\n")
    
    # Run all examples
    example_standard_transformer()
    example_moe_transformer()
    example_kv_cache()
    example_moe_with_cache()
    example_with_attention_mask()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
