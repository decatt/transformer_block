# Transformer Block with KV Cache and Mixture of Experts

A PyTorch implementation of a transformer block with support for:
- **KV Cache**: Efficient key-value caching for autoregressive generation
- **Mixture of Experts (MoE)**: Conditional computation with multiple expert networks
- **Standard Transformer**: Traditional transformer block architecture

## Features

- ✅ Multi-head attention with optional KV caching
- ✅ Mixture of Experts with top-k gating
- ✅ Load balancing loss for MoE
- ✅ Flexible architecture supporting both standard and MoE modes
- ✅ Comprehensive test suite
- ✅ Easy-to-use API

## Installation

```bash
pip install -e .
```

For development with testing dependencies:
```bash
pip install -e ".[dev]"
```

## Quick Start

### Standard Transformer Block

```python
import torch
from transformer_block import TransformerBlock

# Create a standard transformer block
block = TransformerBlock(
    d_model=128,
    n_heads=8,
    d_ff=512,
    dropout=0.1,
    use_moe=False,
)

# Forward pass
x = torch.randn(2, 10, 128)  # (batch_size, seq_len, d_model)
output, _, _ = block(x)
```

### Transformer Block with Mixture of Experts

```python
# Create a transformer block with MoE
block = TransformerBlock(
    d_model=128,
    n_heads=8,
    d_ff=512,
    use_moe=True,
    num_experts=8,
    top_k_experts=2,
)

# Forward pass with auxiliary loss
output, _, aux_loss = block(x, return_aux_loss=True)
```

### Using KV Cache for Efficient Inference

```python
# Create transformer block
block = TransformerBlock(d_model=64, n_heads=4, d_ff=256)
block.eval()

# Create KV cache
cache = block.create_kv_cache(max_batch_size=1, max_seq_len=100)

# Process initial sequence
x1 = torch.randn(1, 5, 64)
output1, _, _ = block(x1, kv_cache=cache, use_cache=True, cache_position=0)

# Process next token (autoregressive)
x2 = torch.randn(1, 1, 64)
output2, _, _ = block(x2, kv_cache=cache, use_cache=True, cache_position=5)
```

### Combining MoE and KV Cache

```python
# Create MoE transformer with cache
block = TransformerBlock(
    d_model=64,
    n_heads=4,
    d_ff=256,
    use_moe=True,
    num_experts=4,
    top_k_experts=2,
)

cache = block.create_kv_cache(max_batch_size=2, max_seq_len=50)
x = torch.randn(2, 10, 64)
output, _, _ = block(x, kv_cache=cache, use_cache=True)
```

## Architecture

### KV Cache
The `KVCache` class stores computed keys and values during attention, enabling efficient autoregressive generation by avoiding recomputation for previous tokens.

### Mixture of Experts
The `MixtureOfExperts` module implements conditional computation:
- A gating network selects top-k experts for each token
- Only selected experts process the input
- Load balancing loss encourages uniform expert usage

### Transformer Block
The `TransformerBlock` combines:
- Multi-head self-attention with optional KV caching
- Feed-forward network or MoE layer
- Layer normalization and residual connections
- Dropout regularization

## API Reference

### TransformerBlock

```python
TransformerBlock(
    d_model: int,              # Model dimension
    n_heads: int,              # Number of attention heads
    d_ff: int,                 # Feed-forward hidden dimension
    dropout: float = 0.1,      # Dropout probability
    use_moe: bool = False,     # Use Mixture of Experts
    num_experts: int = 8,      # Number of experts (if use_moe=True)
    top_k_experts: int = 2,    # Top-k experts to activate (if use_moe=True)
)
```

### KVCache

```python
KVCache(
    max_batch_size: int,       # Maximum batch size
    max_seq_len: int,          # Maximum sequence length
    n_heads: int,              # Number of attention heads
    head_dim: int,             # Dimension per head
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
)
```

### MixtureOfExperts

```python
MixtureOfExperts(
    d_model: int,                      # Model dimension
    d_ff: int,                         # Hidden dimension per expert
    num_experts: int = 8,              # Number of experts
    top_k: int = 2,                    # Number of experts to activate
    dropout: float = 0.1,              # Dropout probability
    expert_capacity_factor: float = 1.25,  # Expert capacity factor
)
```

## Testing

Run the test suite:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest tests/ --cov=transformer_block --cov-report=html
```

## Examples

See `examples.py` for more detailed usage examples:
```bash
python examples.py
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0

## License

This project is open source and available for use.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.