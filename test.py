import math
from typing import Dict, Optional, Tuple, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_block import TransformerBlock

if __name__ == "__main__":

    # 测试基础前向传播
    
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    B, H, Tin, Tout, D = 2, 8, 34, 16, 384
    
    x_in_r = torch.randn(B, H, Tin, D, device=device)
    x_in_i = torch.randn(B, H, Tin, D, device=device)
    x_out_r = torch.randn(B, H, Tout, D, device=device)
    x_out_i = torch.randn(B, H, Tout, D, device=device)
    
    pe_in = 0.01 * torch.randn(B, H, Tin, D, device=device)
    pe_out = 0.01 * torch.randn(B, H, Tout, D, device=device)
    
    block = TransformerBlock(
        n_heads=H,
        d_model=D,
        attn_dropout=0.1,
        ffn_dropout=0.1,
        ffn_mult=4,
        enable_kv_cache=False,
        enable_moe=False,
    ).to(device)
    
    block.train()
    
    y_in_r, y_in_i, y_out_r, y_out_i, aux = block(
        x_in_r, x_in_i, x_out_r, x_out_i, pe_in, pe_out
    )
    
    print(f"Input shapes:  x_in={x_in_r.shape}, x_out={x_out_r.shape}")
    print(f"Output shapes: y_in={y_in_r.shape}, y_out={y_out_r.shape}")
    print(f"Aux losses: {aux}")
    
    # 测试反向传播
    loss = (y_in_r.pow(2).mean() + y_in_i.pow(2).mean() + 
            y_out_r.pow(2).mean() + y_out_i.pow(2).mean())
    loss.backward()
    
    print(f"Loss: {loss.item():.6f}")

    # 测试 KV 缓存以进行增量生成。
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    B, H, D = 2, 4, 32
    Tout = 3
    
    block = TransformerBlock(
        n_heads=H,
        d_model=D,
        attn_dropout=0.0,
        ffn_dropout=0.0,
        ffn_mult=4,
        enable_kv_cache=True,
        enable_moe=False,
        max_cache_len=100,
    ).to(device)
    
    block.eval()
    
    kv_cache = {}
    
    x_out_r = torch.randn(B, H, Tout, D, device=device)
    x_out_i = torch.randn(B, H, Tout, D, device=device)
    pe_out = 0.01 * torch.randn(B, H, Tout, D, device=device)
    
    Tin1 = 5
    x_in_r_1 = torch.randn(B, H, Tin1, D, device=device)
    x_in_i_1 = torch.randn(B, H, Tin1, D, device=device)
    pe_in_1 = 0.01 * torch.randn(B, H, Tin1, D, device=device)
    
    with torch.no_grad():
        y1_in_r, y1_in_i, y1_out_r, y1_out_i, _ = block(
            x_in_r_1, x_in_i_1, x_out_r, x_out_i,
            pe_in_1, pe_out,
            kv_cache=kv_cache,
            use_kv_cache=True,
        )
    
    cache_len_1 = kv_cache["self_in"]["k_r"].shape[2]
    print(f"Input tokens={Tin1}, Cache length={cache_len_1}")

    # 添加更多 tokens
    Tin2 = 3
    x_in_r_2 = torch.randn(B, H, Tin2, D, device=device)
    x_in_i_2 = torch.randn(B, H, Tin2, D, device=device)
    pe_in_2 = 0.01 * torch.randn(B, H, Tin2, D, device=device)
    
    with torch.no_grad():
        y2_in_r, y2_in_i, y2_out_r, y2_out_i, _ = block(
            x_in_r_2, x_in_i_2, x_out_r, x_out_i,
            pe_in_2, pe_out,
            kv_cache=kv_cache,
            use_kv_cache=True,
        )
    
    cache_len_2 = kv_cache["self_in"]["k_r"].shape[2]
    print(f"Input tokens={Tin2}, Cache length={cache_len_2}")
    
    assert cache_len_2 == cache_len_1 + Tin2, "Cache should accumulate!"
    print("✓ Cache accumulation OK")
    
    # Test cache length limiting
    print(f"\nTesting cache limit (max_cache_len=100)...")
    for step in range(3, 25):
        x_in_r_s = torch.randn(B, H, 5, D, device=device)
        x_in_i_s = torch.randn(B, H, 5, D, device=device)
        pe_in_s = 0.01 * torch.randn(B, H, 5, D, device=device)
        
        with torch.no_grad():
            _, _, _, _, _ = block(
                x_in_r_s, x_in_i_s, x_out_r, x_out_i,
                pe_in_s, pe_out,
                kv_cache=kv_cache,
                use_kv_cache=True,
            )
    
    final_cache_len = kv_cache["self_in"]["k_r"].shape[2]
    print(f"Final cache length: {final_cache_len} (should be ≤ 100)")
    assert final_cache_len <= 100, "Cache should be limited!"
    print("✓ Cache limiting OK")
    print()

    # 测试 MoE 负载均衡损失和专家使用情况。
    
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    B, H, Tin, Tout, D = 2, 8, 34, 16, 384
    
    x_in_r = torch.randn(B, H, Tin, D, device=device)
    x_in_i = torch.randn(B, H, Tin, D, device=device)
    x_out_r = torch.randn(B, H, Tout, D, device=device)
    x_out_i = torch.randn(B, H, Tout, D, device=device)
    
    pe_in = 0.01 * torch.randn(B, H, Tin, D, device=device)
    pe_out = 0.01 * torch.randn(B, H, Tout, D, device=device)
    
    block = TransformerBlock(
        n_heads=H,
        d_model=D,
        attn_dropout=0.0,
        ffn_dropout=0.0,
        ffn_mult=4,
        enable_kv_cache=False,
        enable_moe=True,
        moe_num_experts=8,
        moe_top_k=2,
    ).to(device)
    
    block.train()
    
    y_in_r, y_in_i, y_out_r, y_out_i, aux = block(
        x_in_r, x_in_i, x_out_r, x_out_i, pe_in, pe_out,
        use_moe=True,
    )
    
    print(f"Output shapes: y_in={y_in_r.shape}, y_out={y_out_r.shape}")
    print(f"MoE aux losses: {list(aux.keys())}")
    
    if 'moe_loss_in' in aux:
        print(f"  moe_loss_in: {aux['moe_loss_in'].item():.6f}")
    if 'moe_loss_out' in aux:
        print(f"  moe_loss_out: {aux['moe_loss_out'].item():.6f}")
    
    # Check expert usage
    expert_counts_in = block.ffn_in.expert_counts
    expert_counts_out = block.ffn_out.expert_counts
    
    print(f"\nExpert usage (x_in FFN): {expert_counts_in.cpu().numpy()}")
    print(f"Expert usage (x_out FFN): {expert_counts_out.cpu().numpy()}")
    
    total_tokens_in = B * H * Tin
    total_tokens_out = B * H * Tout
    print(f"Total tokens: x_in={total_tokens_in}, x_out={total_tokens_out}")
    print("✓ MoE forward OK")
    
    # Test backward with aux loss
    main_loss = (y_in_r.pow(2).mean() + y_in_i.pow(2).mean() + 
                 y_out_r.pow(2).mean() + y_out_i.pow(2).mean())
    
    total_loss = main_loss
    if 'moe_loss_in' in aux:
        total_loss = total_loss + 0.01 * aux['moe_loss_in']
    if 'moe_loss_out' in aux:
        total_loss = total_loss + 0.01 * aux['moe_loss_out']
    
    total_loss.backward()
    print("✓ Backward with aux losses OK")
    print()

    # 测试数值稳定性
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B, H, T, D = 2, 8, 34, 384

    # 测试非常小的值
    print("Testing with small values (1e-6)...")
    x_r = 1e-6 * torch.randn(B, H, T, D, device=device)
    x_i = 1e-6 * torch.randn(B, H, T, D, device=device)
    
    block = TransformerBlock(n_heads=H, d_model=D).to(device)
    block.eval()
    
    with torch.no_grad():
        y_in_r, y_in_i, y_out_r, y_out_i, _ = block(x_r, x_i, x_r, x_i)
    
    assert not torch.isnan(y_in_r).any(), "NaN detected in output (small values)!"
    assert not torch.isinf(y_in_r).any(), "Inf detected in output (small values)!"
    print("✓ Small values OK")

    # 测试大值
    print("Testing with large values (1e3)...")
    x_r = 1e3 * torch.randn(B, H, T, D, device=device)
    x_i = 1e3 * torch.randn(B, H, T, D, device=device)
    
    with torch.no_grad():
        y_in_r, y_in_i, y_out_r, y_out_i, _ = block(x_r, x_i, x_r, x_i)
    
    assert not torch.isnan(y_in_r).any(), "NaN detected in output (large values)!"
    assert not torch.isinf(y_in_r).any(), "Inf detected in output (large values)!"
    print("✓ Large values OK")

    # 测试相位保留
    print("Testing phase preservation...")
    x_r = torch.randn(B, H, T, D, device=device)
    x_i = torch.randn(B, H, T, D, device=device)
    
    phase_in = torch.atan2(x_i, x_r + 1e-8)
    
    with torch.no_grad():
        y_in_r, y_in_i, y_out_r, y_out_i, _ = block(x_r, x_i, x_r, x_i)
    
    phase_out = torch.atan2(y_in_i, y_in_r + 1e-8)

    phase_std = phase_out.std().item()
    print(f"Output phase std: {phase_std:.6f}")
    assert phase_std > 0.01, "Phase appears collapsed!"
    print("✓ Phase preservation OK")
    print()

    print("All tests passed! ✓")