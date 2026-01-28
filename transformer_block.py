import math
from typing import Dict, Optional, Tuple, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from complex_layer import ComplexHeadwiseLayerNorm, ComplexAttention, ComplexMoE, ComplexSwiGLUFFN, ComplexDropout

class TransformerBlock(nn.Module):
    def __init__(
        self,
        n_heads: int,
        d_model: int,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        ffn_mult: int = 2,
        enable_kv_cache: bool = False,
        enable_moe: bool = False,
        moe_num_experts: int = 8,
        moe_top_k: int = 2,
        max_cache_len: int = 4096,
    ):
        super().__init__()
        self.n_heads = int(n_heads)
        self.d_model = int(d_model)
        self.enable_kv_cache = bool(enable_kv_cache)
        self.enable_moe = bool(enable_moe)

        # Pre-norms for each residual path
        self.ln_in_self = ComplexHeadwiseLayerNorm(n_heads, d_model)
        self.ln_out_cross_q = ComplexHeadwiseLayerNorm(n_heads, d_model)
        self.ln_in_cross_kv = ComplexHeadwiseLayerNorm(n_heads, d_model)
        self.ln_in_ffn = ComplexHeadwiseLayerNorm(n_heads, d_model)
        self.ln_out_ffn = ComplexHeadwiseLayerNorm(n_heads, d_model)

        # Attentions
        self.self_attn_in = ComplexAttention(d_model, dropout_p=attn_dropout, max_cache_len=max_cache_len)
        self.cross_attn = ComplexAttention(d_model, dropout_p=attn_dropout, max_cache_len=max_cache_len)

        # FFN / MoE
        d_hidden = ffn_mult * d_model
        if self.enable_moe:
            self.ffn_in = ComplexMoE(d_model, d_hidden, moe_num_experts, moe_top_k, dropout_p=ffn_dropout)
            self.ffn_out = ComplexMoE(d_model, d_hidden, moe_num_experts, moe_top_k, dropout_p=ffn_dropout)
        else:
            self.ffn_in = ComplexSwiGLUFFN(d_model, d_hidden, dropout_p=ffn_dropout)
            self.ffn_out = ComplexSwiGLUFFN(d_model, d_hidden, dropout_p=ffn_dropout)

        self.resid_drop = ComplexDropout(ffn_dropout)

    @staticmethod
    def _add_pe_to_real_only(
        x_r: torch.Tensor, 
        x_i: torch.Tensor, 
        pe: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if pe is None:
            return x_r, x_i
        return x_r + pe, x_i

    def forward(
        self,
        x_in_r: torch.Tensor,
        x_in_i: torch.Tensor,
        x_out_r: torch.Tensor,
        x_out_i: torch.Tensor,
        pe_in: Optional[torch.Tensor] = None,
        pe_out: Optional[torch.Tensor] = None,
        kv_cache: Optional[Dict[str, Any]] = None,
        use_kv_cache: Optional[bool] = None,
        use_moe: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if use_kv_cache is None:
            use_kv_cache = self.enable_kv_cache
        if use_moe is None:
            use_moe = self.enable_moe

        aux_losses = {}

        x_in_r, x_in_i = self._add_pe_to_real_only(x_in_r, x_in_i, pe_in)
        x_out_r, x_out_i = self._add_pe_to_real_only(x_out_r, x_out_i, pe_out)

        # ---- x_in: Self-attention ----
        n_in_r, n_in_i = self.ln_in_self(x_in_r, x_in_i)
        
        sa_r, sa_i = self.self_attn_in(
            q_in_r=n_in_r, 
            q_in_i=n_in_i,
            kv_in_r=n_in_r, 
            kv_in_i=n_in_i,
            use_kv_cache=bool(use_kv_cache),
            kv_cache=kv_cache,
            cache_slot="self_in",
        )
        
        sa_r, sa_i = self.resid_drop(sa_r, sa_i)
        x_in_r = x_in_r + sa_r
        x_in_i = x_in_i + sa_i

        # ---- x_out: Cross-attention (Q from x_out, KV from x_in) ----
        nq_r, nq_i = self.ln_out_cross_q(x_out_r, x_out_i)
        nkv_r, nkv_i = self.ln_in_cross_kv(x_in_r, x_in_i)

        ca_r, ca_i = self.cross_attn(
            q_in_r=nq_r, 
            q_in_i=nq_i,
            kv_in_r=nkv_r, 
            kv_in_i=nkv_i,
            use_kv_cache=bool(use_kv_cache),
            kv_cache=kv_cache,
            cache_slot="cross_in",
        )
        
        ca_r, ca_i = self.resid_drop(ca_r, ca_i)
        x_out_r = x_out_r + ca_r
        x_out_i = x_out_i + ca_i

        # ---- x_in: FFN / MoE ----
        nf_in_r, nf_in_i = self.ln_in_ffn(x_in_r, x_in_i)
        
        if use_moe:
            f_in_r, f_in_i, moe_loss_in = self.ffn_in(nf_in_r, nf_in_i)
            if moe_loss_in is not None:
                aux_losses['moe_loss_in'] = moe_loss_in
        else:
            f_in_r, f_in_i = self.ffn_in(nf_in_r, nf_in_i)
        
        f_in_r, f_in_i = self.resid_drop(f_in_r, f_in_i)
        x_in_r = x_in_r + f_in_r
        x_in_i = x_in_i + f_in_i

        # ---- x_out: FFN / MoE ----
        nf_out_r, nf_out_i = self.ln_out_ffn(x_out_r, x_out_i)
        
        if use_moe:
            f_out_r, f_out_i, moe_loss_out = self.ffn_out(nf_out_r, nf_out_i)
            if moe_loss_out is not None:
                aux_losses['moe_loss_out'] = moe_loss_out
        else:
            f_out_r, f_out_i = self.ffn_out(nf_out_r, nf_out_i)
        
        f_out_r, f_out_i = self.resid_drop(f_out_r, f_out_i)
        x_out_r = x_out_r + f_out_r
        x_out_i = x_out_i + f_out_i

        return x_in_r, x_in_i, x_out_r, x_out_i, aux_losses
