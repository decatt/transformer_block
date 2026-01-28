import math
from typing import Dict, Optional, Tuple, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def complex_mul(
    a_r: torch.Tensor, 
    a_i: torch.Tensor, 
    b_r: torch.Tensor, 
    b_i: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    return a_r * b_r - a_i * b_i, a_r * b_i + a_i * b_r


def complex_magnitude(x_r: torch.Tensor, x_i: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(x_r ** 2 + x_i ** 2 + eps)


class ComplexDropout(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = float(p)

    def forward(
        self, 
        x_r: torch.Tensor, 
        x_i: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.p <= 0.0:
            return x_r, x_i
        mask = (torch.rand_like(x_r) > self.p).to(x_r.dtype) / (1.0 - self.p)
        return x_r * mask, x_i * mask


class ComplexLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        self.w_r = nn.Parameter(torch.empty(out_features, in_features))
        self.w_i = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.b_r = nn.Parameter(torch.zeros(out_features))
            self.b_i = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('b_r', None)
            self.register_parameter('b_i', None)

        self._init_weights()

    def _init_weights(self):
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        
        nn.init.normal_(self.w_r, mean=0.0, std=std)
        nn.init.normal_(self.w_i, mean=0.0, std=std)
        
        scale = 1.0 / math.sqrt(2.0)
        self.w_r.data.mul_(scale)
        self.w_i.data.mul_(scale)

    def forward(
        self, 
        x_r: torch.Tensor, 
        x_i: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        y_r = torch.matmul(x_r, self.w_r.t()) - torch.matmul(x_i, self.w_i.t())
        y_i = torch.matmul(x_i, self.w_r.t()) + torch.matmul(x_r, self.w_i.t())
        
        if self.b_r is not None:
            y_r = y_r + self.b_r
            y_i = y_i + self.b_i
            
        return y_r, y_i


class HeadwiseLayerNorm(nn.Module):
    def __init__(self, n_heads: int, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.n_heads = int(n_heads)
        self.d_model = int(d_model)
        self.eps = float(eps)

        self.weight = nn.Parameter(torch.ones(n_heads, d_model))
        self.bias = nn.Parameter(torch.zeros(n_heads, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, T, D]
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        # Apply affine per head: [1, H, 1, D]
        return x_norm * self.weight[None, :, None, :] + self.bias[None, :, None, :]


class ComplexHeadwiseLayerNorm(nn.Module):    
    def __init__(self, n_heads: int, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.ln_r = HeadwiseLayerNorm(n_heads, d_model, eps)
        self.ln_i = HeadwiseLayerNorm(n_heads, d_model, eps)

    def forward(
        self, 
        x_r: torch.Tensor, 
        x_i: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.ln_r(x_r), self.ln_i(x_i)


class ComplexSwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, dropout_p: float = 0.0):
        super().__init__()
        self.up_gate = ComplexLinear(d_model, d_hidden, bias=True)
        self.up_value = ComplexLinear(d_model, d_hidden, bias=True)
        self.down = ComplexLinear(d_hidden, d_model, bias=True)
        self.drop = ComplexDropout(dropout_p)

    def forward(
        self, 
        x_r: torch.Tensor, 
        x_i: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gate_r, gate_i = self.up_gate(x_r, x_i)
        gate_mag = complex_magnitude(gate_r, gate_i)
        gate_activated = F.silu(gate_mag)  # [B, H, T, D_hidden]
        
        val_r, val_i = self.up_value(x_r, x_i)
        
        h_r = gate_activated * val_r
        h_i = gate_activated * val_i
        
        y_r, y_i = self.down(h_r, h_i)
        y_r, y_i = self.drop(y_r, y_i)
        
        return y_r, y_i


class ComplexMoE(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        num_experts: int,
        top_k: int,
        dropout_p: float = 0.0,
        load_balance_weight: float = 0.01,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.d_hidden = int(d_hidden)
        self.num_experts = int(num_experts)
        self.top_k = min(int(top_k), num_experts)
        self.load_balance_weight = float(load_balance_weight)

        self.router = nn.Linear(2 * d_model, num_experts, bias=True)
        
        self.experts = nn.ModuleList([
            ComplexSwiGLUFFN(d_model, d_hidden, dropout_p) 
            for _ in range(num_experts)
        ])
        
        self.register_buffer('expert_counts', torch.zeros(num_experts))

    def compute_load_balance_loss(
        self, 
        router_probs: torch.Tensor, 
        expert_mask: torch.Tensor
    ) -> torch.Tensor:
        prob_per_expert = router_probs.mean(dim=0)  # [E]
        
        frac_per_expert = expert_mask.float().mean(dim=0)  # [E]
        
        loss = self.num_experts * (prob_per_expert * frac_per_expert).sum()
        return loss

    def forward(
        self, 
        x_r: torch.Tensor, 
        x_i: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        B, H, T, D = x_r.shape
        N = B * H * T

        xr_flat = x_r.reshape(N, D)
        xi_flat = x_i.reshape(N, D)

        router_input = torch.cat([xr_flat, xi_flat], dim=-1)  # [N, 2D]
        logits = self.router(router_input)  # [N, E]
        
        topk_logits, topk_indices = torch.topk(logits, k=self.top_k, dim=-1)  # [N, K]
        topk_weights = F.softmax(topk_logits, dim=-1)  # [N, K]
        
        out_r = torch.zeros_like(xr_flat)
        out_i = torch.zeros_like(xi_flat)
        
        expert_mask = torch.zeros_like(logits)  # [N, E] for load balance
        
        for expert_id in range(self.num_experts):
            expert_positions = (topk_indices == expert_id)  # [N, K]
            
            if not expert_positions.any():
                continue

            token_mask = expert_positions.any(dim=-1)  # [N]
            token_indices = torch.nonzero(token_mask, as_tuple=False).squeeze(-1)
            
            if token_indices.numel() == 0:
                continue

            weights_for_expert = (topk_weights * expert_positions.float()).sum(dim=-1, keepdim=True)  # [N, 1]
            weights_selected = weights_for_expert[token_indices]  # [n_selected, 1]

            xr_selected = xr_flat[token_indices]
            xi_selected = xi_flat[token_indices]

            yr_expert, yi_expert = self.experts[expert_id](xr_selected, xi_selected)

            yr_weighted = yr_expert * weights_selected
            yi_weighted = yi_expert * weights_selected
            
            out_r.index_add_(0, token_indices, yr_weighted)
            out_i.index_add_(0, token_indices, yi_weighted)

            expert_mask[token_indices, expert_id] = 1.0

            if self.training:
                self.expert_counts[expert_id] += token_indices.numel()

        load_balance_loss = None
        if self.training:
            router_probs = F.softmax(logits, dim=-1)
            load_balance_loss = self.compute_load_balance_loss(router_probs, expert_mask)

        out_r = out_r.reshape(B, H, T, D)
        out_i = out_i.reshape(B, H, T, D)
        
        return out_r, out_i, load_balance_loss

class ComplexAttention(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        dropout_p: float = 0.0,
        max_cache_len: int = 4096,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.max_cache_len = int(max_cache_len)
        
        self.q_proj = ComplexLinear(d_model, d_model, bias=True)
        self.k_proj = ComplexLinear(d_model, d_model, bias=True)
        self.v_proj = ComplexLinear(d_model, d_model, bias=True)
        
        self.out_proj = ComplexLinear(d_model, d_model, bias=True)

        self.gate_proj = nn.Linear(d_model, d_model, bias=True)
        
        self.drop_attn = nn.Dropout(dropout_p)
        self.drop_out = ComplexDropout(dropout_p)

    @staticmethod
    def _get_or_create_cache_slot(
        kv_cache: Dict[str, Any], 
        slot: str
    ) -> Dict[str, torch.Tensor]:
        if slot not in kv_cache or kv_cache[slot] is None:
            kv_cache[slot] = {}
        return kv_cache[slot]

    def _limit_cache_length(
        self, 
        k_r: torch.Tensor, 
        k_i: torch.Tensor,
        v_r: torch.Tensor, 
        v_i: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if k_r.shape[2] > self.max_cache_len:
            k_r = k_r[:, :, -self.max_cache_len:, :]
            k_i = k_i[:, :, -self.max_cache_len:, :]
            v_r = v_r[:, :, -self.max_cache_len:, :]
            v_i = v_i[:, :, -self.max_cache_len:, :]
        return k_r, k_i, v_r, v_i

    def forward(
        self,
        q_in_r: torch.Tensor,
        q_in_i: torch.Tensor,
        kv_in_r: torch.Tensor,
        kv_in_i: torch.Tensor,
        use_kv_cache: bool = False,
        kv_cache: Optional[Dict[str, Any]] = None,
        cache_slot: str = "kv",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, Tq, D = q_in_r.shape

        q_r, q_i = self.q_proj(q_in_r, q_in_i)

        k_r, k_i = self.k_proj(kv_in_r, kv_in_i)
        v_r, v_i = self.v_proj(kv_in_r, kv_in_i)

        if use_kv_cache and kv_cache is not None:
            slot = self._get_or_create_cache_slot(kv_cache, cache_slot)
            
            if "k_r" in slot:
                k_r = torch.cat([slot["k_r"], k_r], dim=2)
                k_i = torch.cat([slot["k_i"], k_i], dim=2)
                v_r = torch.cat([slot["v_r"], v_r], dim=2)
                v_i = torch.cat([slot["v_i"], v_i], dim=2)

            k_r, k_i, v_r, v_i = self._limit_cache_length(k_r, k_i, v_r, v_i)

            slot["k_r"] = k_r.detach()
            slot["k_i"] = k_i.detach()
            slot["v_r"] = v_r.detach()
            slot["v_i"] = v_i.detach()

        Tk = k_r.shape[2]
        scale = 1.0 / math.sqrt(D)

        scores = (
            torch.einsum("bhqd,bhkd->bhqk", q_r, k_r) +
            torch.einsum("bhqd,bhkd->bhqk", q_i, k_i)
        ) * scale

        attn = torch.softmax(scores, dim=-1)
        attn = self.drop_attn(attn)

        ctx_r = torch.einsum("bhqk,bhkd->bhqd", attn, v_r)
        ctx_i = torch.einsum("bhqk,bhkd->bhqd", attn, v_i)

        out_r, out_i = self.out_proj(ctx_r, ctx_i)

        out_mag = complex_magnitude(out_r, out_i)
        gate = torch.sigmoid(self.gate_proj(out_mag))  # [B, H, Tq, D], range [0, 1]
        
        out_r = out_r * gate
        out_i = out_i * gate
        
        out_r, out_i = self.drop_out(out_r, out_i)
        
        return out_r, out_i