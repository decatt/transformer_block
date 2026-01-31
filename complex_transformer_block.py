import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


# ===========================================================================
# Shape convention (贯穿全文)
# ===========================================================================
#
#   native  layout:  [b, p, beam, fea]          ← FFN / Norm / Residual
#   attn    layout:  [b, head, p, attn_dim]     ← Attention (Q·K, attn·V)
#   PE  always lives in attn layout.
#
#   attn_dim = beam * fea // head
#
#   native → attn :  .reshape(b, p, head, attn_dim).permute(0,2,1,3)
#   attn   → native:  .permute(0,2,1,3).reshape(b, p, beam, fea)
#
#   defaults:  b=4, p_in=34, p_out=16, beam=32, fea=96, head=8
#              → attn_dim = 32*96 // 8 = 384
# ===========================================================================


# ---------------------------------------------------------------------------
# ComplexLinear  — 作用在 last dim，对前缀维广播
# ---------------------------------------------------------------------------
class ComplexLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight_r = nn.Parameter(
            torch.empty(out_features, in_features).normal_(std=1.0 / math.sqrt(in_features))
        )
        self.weight_i = nn.Parameter(
            torch.empty(out_features, in_features).normal_(std=1.0 / math.sqrt(in_features))
        )
        if bias:
            self.bias_r = nn.Parameter(torch.zeros(out_features))
            self.bias_i = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias_r', None)
            self.register_parameter('bias_i', None)

    def forward(self, x_r, x_i):
        o_r = F.linear(x_r, self.weight_r) - F.linear(x_i, self.weight_i)
        o_i = F.linear(x_r, self.weight_i) + F.linear(x_i, self.weight_r)
        if self.bias_r is not None:
            o_r = o_r + self.bias_r
            o_i = o_i + self.bias_i
        return o_r, o_i


# ---------------------------------------------------------------------------
# ComplexRMSNorm — 对 last dim 归一化
# ---------------------------------------------------------------------------
class ComplexRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x_r, x_i):
        rms = torch.sqrt((x_r ** 2 + x_i ** 2).mean(dim=-1, keepdim=True) + self.eps)
        return x_r / rms * self.weight, x_i / rms * self.weight


# ---------------------------------------------------------------------------
# ComplexSwiGLU — SiLU gate on magnitude, keeps phase
# ---------------------------------------------------------------------------
class ComplexSwiGLU(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = ComplexLinear(dim, dim, bias=False)
        self.up   = ComplexLinear(dim, dim, bias=False)

    def forward(self, x_r, x_i):
        g_r, g_i = self.gate(x_r, x_i)
        u_r, u_i = self.up(x_r, x_i)

        mag   = torch.sqrt(g_r ** 2 + g_i ** 2 + 1e-8)
        phase = torch.atan2(g_i, g_r)
        mag   = F.silu(mag)
        g_r, g_i = mag * torch.cos(phase), mag * torch.sin(phase)

        return (g_r * u_r - g_i * u_i,
                g_r * u_i + g_i * u_r)


# ===========================================================================
# MoE  —  native layout  [b, p, beam, fea]
# ===========================================================================

class ComplexMoEExpert(nn.Module):
    def __init__(self, fea: int, hidden: int):
        super().__init__()
        self.fc1 = ComplexLinear(fea, hidden)
        self.act = ComplexSwiGLU(hidden)
        self.fc2 = ComplexLinear(hidden, fea)

    def forward(self, x_r, x_i):
        h_r, h_i = self.fc1(x_r, x_i)
        h_r, h_i = self.act(h_r, h_i)
        return self.fc2(h_r, h_i)


class ComplexMoE(nn.Module):
    """
    Top-k MoE.  Only selected experts are actually executed per token.

    Router works on fea dim.  Routing decision is per (b, p, beam) position.
    Tokens are flattened to [N, fea], routed, then only the selected subset
    is passed through each expert  →  no wasted FLOPs.
    """
    def __init__(self, fea: int, num_experts: int = 8, top_k: int = 2, hidden: int = None):
        super().__init__()
        self.num_experts = num_experts
        self.top_k       = top_k
        hidden           = hidden or fea * 4

        self.experts = nn.ModuleList([ComplexMoEExpert(fea, hidden) for _ in range(num_experts)])
        self.router  = ComplexLinear(fea, num_experts, bias=False)

    def forward(self, x_r, x_i):
        orig_shape = x_r.shape                          # [b, p, beam, fea]
        fea  = orig_shape[-1]
        N    = x_r[..., 0].numel()                      # b*p*beam
        xr_f = x_r.reshape(N, fea)
        xi_f = x_i.reshape(N, fea)

        # --- router (magnitude of complex logits) ---
        lr, li = self.router(xr_f, xi_f)                # [N, num_experts]
        logits = torch.sqrt(lr ** 2 + li ** 2)

        weights, indices = torch.topk(logits, self.top_k, dim=-1)  # [N, k]
        weights = F.softmax(weights, dim=-1)

        out_r = torch.zeros_like(xr_f)
        out_i = torch.zeros_like(xi_f)

        # --- dispatch only selected tokens to each expert ---
        for ki in range(self.top_k):
            idx = indices[:, ki]       # [N]
            w   = weights[:, ki]       # [N]

            for eid in range(self.num_experts):
                mask = (idx == eid)
                if not mask.any():
                    continue
                # extract subset, run expert, scatter back
                er, ei = self.experts[eid](xr_f[mask], xi_f[mask])
                out_r[mask] = out_r[mask] + er * w[mask].unsqueeze(-1)
                out_i[mask] = out_i[mask] + ei * w[mask].unsqueeze(-1)

        return out_r.reshape(orig_shape), out_i.reshape(orig_shape)


# ===========================================================================
# Differential Attention  —  attn layout  [b, head, p, attn_dim]
# ===========================================================================

class ComplexDifferentialAttention(nn.Module):
    """
    Differential Attention in complex domain.

    Q → 2*attn_dim  (split into q1, q2 for two attention streams)
    K → attn_dim    (independent projection)
    V → attn_dim    (independent projection)

    λ re-parameterization  (shared across heads):
        λ₁ = exp(⟨λ_q1, λ_k1⟩)
        λ₂ = exp(⟨λ_q2, λ_k2⟩)
        λ  = sigmoid(λ₁ − λ₂ + λ_init)

    SubLN: RMSNorm on [a1 ‖ a2] before differential.
    Gate:  complex multiply with gate_proj(q).
    """
    def __init__(self, attn_dim: int, num_heads: int, depth: int = 1):
        super().__init__()
        self.attn_dim = attn_dim
        self.scale    = attn_dim ** -0.5

        self.q_proj    = ComplexLinear(attn_dim, attn_dim * 2)
        self.k_proj    = ComplexLinear(attn_dim, attn_dim)
        self.v_proj    = ComplexLinear(attn_dim, attn_dim)
        self.gate_proj = ComplexLinear(attn_dim, attn_dim)
        self.out_proj  = ComplexLinear(attn_dim, attn_dim)

        # λ params  [attn_dim], shared across heads
        self.lambda_q1 = nn.Parameter(torch.zeros(attn_dim).normal_(std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(attn_dim).normal_(std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(attn_dim).normal_(std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(attn_dim).normal_(std=0.1))
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * depth)

        self.subln = ComplexRMSNorm(2 * attn_dim)

    # ------------------------------------------------------------------
    def forward(
        self,
        q_r, q_i,           # [b, head, p_q, attn_dim]
        k_r, k_i,           # [b, head, p_k, attn_dim]   (raw, before k_proj)
        v_r, v_i,           # [b, head, p_k, attn_dim]   (raw, before v_proj)
        pe_q_r, pe_q_i,     # [b, head, p_q, attn_dim]
        pe_k_r, pe_k_i,     # [b, head, p_k, attn_dim]
        kv_cache: Optional[Tuple] = None,
        use_cache: bool = False
    ):
        # ---------- independent projections ----------
        qp_r, qp_i = self.q_proj(q_r, q_i)     # [b,h,p_q, 2*ad]
        kp_r, kp_i = self.k_proj(k_r, k_i)     # [b,h,p_k, ad]
        vp_r, vp_i = self.v_proj(v_r, v_i)     # [b,h,p_k, ad]  ← independent from K

        # ---------- add PE ----------
        qp_r = qp_r + torch.cat([pe_q_r, pe_q_r], dim=-1)
        qp_i = qp_i + torch.cat([pe_q_i, pe_q_i], dim=-1)
        kp_r = kp_r + pe_k_r
        kp_i = kp_i + pe_k_i

        # ---------- KV cache ----------
        new_cache = None
        if use_cache:
            if kv_cache is not None:
                ck_r, ck_i, cv_r, cv_i = kv_cache
                kp_r = torch.cat([ck_r, kp_r], dim=2)
                kp_i = torch.cat([ck_i, kp_i], dim=2)
                vp_r = torch.cat([cv_r, vp_r], dim=2)
                vp_i = torch.cat([cv_i, vp_i], dim=2)
            new_cache = (kp_r, kp_i, vp_r, vp_i)

        # ---------- split Q ----------
        q1_r, q2_r = qp_r.chunk(2, dim=-1)
        q1_i, q2_i = qp_i.chunk(2, dim=-1)

        # ---------- attention weights ----------
        def _attn_w(qr, qi, kr, ki):
            s_r = torch.matmul(qr, kr.transpose(-2,-1)) + torch.matmul(qi, ki.transpose(-2,-1))
            s_i = torch.matmul(qi, kr.transpose(-2,-1)) - torch.matmul(qr, ki.transpose(-2,-1))
            return F.softmax(
                torch.sqrt(s_r**2 + s_i**2 + 1e-8) * self.scale,
                dim=-1, dtype=torch.float32
            ).type_as(s_r)

        w1 = _attn_w(q1_r, q1_i, kp_r, kp_i)
        w2 = _attn_w(q2_r, q2_i, kp_r, kp_i)

        # ---------- weighted V ----------
        a1_r, a1_i = torch.matmul(w1, vp_r), torch.matmul(w1, vp_i)
        a2_r, a2_i = torch.matmul(w2, vp_r), torch.matmul(w2, vp_i)

        # ---------- SubLN ----------
        cat_r, cat_i = torch.cat([a1_r, a2_r], dim=-1), torch.cat([a1_i, a2_i], dim=-1)
        cat_r, cat_i = self.subln(cat_r, cat_i)
        a1_r, a2_r = cat_r.chunk(2, dim=-1)
        a1_i, a2_i = cat_i.chunk(2, dim=-1)

        # ---------- λ ----------
        lam1 = torch.exp((self.lambda_q1 * self.lambda_k1).sum())
        lam2 = torch.exp((self.lambda_q2 * self.lambda_k2).sum())
        lam  = torch.sigmoid(lam1 - lam2 + self.lambda_init)

        # ---------- differential ----------
        out_r = a1_r - lam * a2_r
        out_i = a1_i - lam * a2_i

        # ---------- gate (complex multiply) ----------
        g_r, g_i = self.gate_proj(q_r, q_i)
        out_r, out_i = (g_r * out_r - g_i * out_i,
                        g_r * out_i + g_i * out_r)

        # ---------- output projection ----------
        out_r, out_i = self.out_proj(out_r, out_i)

        return out_r, out_i, new_cache


# ===========================================================================
# TransformerBlock
# ===========================================================================

class TransformerBlock(nn.Module):
    """
    复数 Transformer Block — Differential Attention + optional MoE + optional KV-cache.

    ┌─────────────────────────────────────────────────────────┐
    │  native  [b, p, beam, fea]   ←  FFN / Norm / Residual   │
    │               │  ▲                                       │
    │    _to_attn   │  │  _from_attn                           │
    │               ▼  │                                       │
    │  attn  [b, head, p, attn_dim]  ←  DifferentialAttention │
    └─────────────────────────────────────────────────────────┘

    x_in  stream:  self-attn  →  cross-attn (Q=x_in, KV=x_out)  →  FFN
    x_out stream:  self-attn  →  FFN

    Constraint: (beam * fea) % num_heads == 0
    Defaults:   beam=32  fea=96  head=8  →  attn_dim = 384
    """
    def __init__(
        self,
        beam: int        = 32,
        fea: int         = 96,
        num_heads: int   = 8,
        use_moe: bool    = True,
        num_experts: int = 8,
        top_k: int       = 2,
        use_kv_cache: bool = False,
        depth: int       = 1,
    ):
        super().__init__()
        assert (beam * fea) % num_heads == 0, \
            f"beam*fea ({beam}*{fea}={beam*fea}) must be divisible by num_heads ({num_heads})"

        self.beam      = beam
        self.fea       = fea
        self.num_heads = num_heads
        self.attn_dim  = beam * fea // num_heads
        self.use_moe   = use_moe
        self.use_kv_cache = use_kv_cache

        ad = self.attn_dim

        # --- Attention (operate on attn_dim) ---
        self.self_attn_in  = ComplexDifferentialAttention(ad, num_heads, depth)
        self.cross_attn    = ComplexDifferentialAttention(ad, num_heads, depth)
        self.self_attn_out = ComplexDifferentialAttention(ad, num_heads, depth)

        # --- Norms (operate on fea, native layout) ---
        self.norm1_in  = ComplexRMSNorm(fea)   # after self-attn  x_in
        self.norm2_in  = ComplexRMSNorm(fea)   # after cross-attn x_in
        self.norm3_in  = ComplexRMSNorm(fea)   # after FFN        x_in
        self.norm1_out = ComplexRMSNorm(fea)   # after self-attn  x_out
        self.norm2_out = ComplexRMSNorm(fea)   # after FFN        x_out

        # --- FFN / MoE (operate on fea, native layout) ---
        if use_moe:
            self.ffn_in  = ComplexMoE(fea, num_experts, top_k)
            self.ffn_out = ComplexMoE(fea, num_experts, top_k)
        else:
            self.ffn_in  = nn.ModuleDict({
                'fc1': ComplexLinear(fea, fea * 4),
                'act': ComplexSwiGLU(fea * 4),
                'fc2': ComplexLinear(fea * 4, fea),
            })
            self.ffn_out = nn.ModuleDict({
                'fc1': ComplexLinear(fea, fea * 4),
                'act': ComplexSwiGLU(fea * 4),
                'fc2': ComplexLinear(fea * 4, fea),
            })

        # --- KV caches ---
        self.kv_cache_self_in  = None
        self.kv_cache_cross    = None
        self.kv_cache_self_out = None

    # ------------------------------------------------------------------
    # layout helpers
    # ------------------------------------------------------------------
    def _to_attn(self, x_r, x_i):
        """[b, p, beam, fea] → [b, head, p, attn_dim]"""
        b, p = x_r.shape[:2]
        h, ad = self.num_heads, self.attn_dim
        return (x_r.reshape(b, p, h, ad).permute(0, 2, 1, 3),
                x_i.reshape(b, p, h, ad).permute(0, 2, 1, 3))

    def _from_attn(self, x_r, x_i):
        """[b, head, p, attn_dim] → [b, p, beam, fea]"""
        b, _, p, _ = x_r.shape
        return (x_r.permute(0, 2, 1, 3).reshape(b, p, self.beam, self.fea),
                x_i.permute(0, 2, 1, 3).reshape(b, p, self.beam, self.fea))

    # ------------------------------------------------------------------
    def _run_ffn(self, ffn, x_r, x_i):
        if self.use_moe:
            return ffn(x_r, x_i)
        h_r, h_i = ffn['fc1'](x_r, x_i)
        h_r, h_i = ffn['act'](h_r, h_i)
        return ffn['fc2'](h_r, h_i)

    # ------------------------------------------------------------------
    def forward(
        self,
        x_in_r,  x_in_i,     # [b, p_in,  beam, fea]
        x_out_r, x_out_i,    # [b, p_out, beam, fea]
        pe_in_r, pe_in_i,    # [b, head, p_in,  attn_dim]
        pe_out_r, pe_out_i,  # [b, head, p_out, attn_dim]
        use_cache: bool = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if use_cache is None:
            use_cache = self.use_kv_cache

        # ============================================================
        # x_in stream
        # ============================================================

        # --- 1. Self-attention ---
        res_r, res_i = x_in_r, x_in_i
        q_r, q_i = self._to_attn(x_in_r, x_in_i)
        # Q / K / V 都从同一个 x_in 进入，但在 DiffAttn 内部各自独立投影

        a_r, a_i, self.kv_cache_self_in = self.self_attn_in(
            q_r, q_i,           # → q_proj
            q_r, q_i,           # → k_proj (独立)
            q_r, q_i,           # → v_proj (独立)
            pe_in_r,  pe_in_i,
            pe_in_r,  pe_in_i,
            kv_cache=self.kv_cache_self_in if use_cache else None,
            use_cache=use_cache,
        )

        a_r, a_i = self._from_attn(a_r, a_i)
        a_r, a_i = self.norm1_in(a_r, a_i)
        x_in_r = res_r + a_r
        x_in_i = res_i + a_i

        # --- 2. Cross-attention: Q = x_in,  K & V = x_out ---
        res_r, res_i = x_in_r, x_in_i
        q_r,  q_i  = self._to_attn(x_in_r,  x_in_i)
        kv_r, kv_i = self._to_attn(x_out_r, x_out_i)

        a_r, a_i, self.kv_cache_cross = self.cross_attn(
            q_r,  q_i,          # → q_proj
            kv_r, kv_i,         # → k_proj
            kv_r, kv_i,         # → v_proj (独立于 k_proj)
            pe_in_r,  pe_in_i,
            pe_out_r, pe_out_i,
            kv_cache=self.kv_cache_cross if use_cache else None,
            use_cache=use_cache,
        )

        a_r, a_i = self._from_attn(a_r, a_i)
        a_r, a_i = self.norm2_in(a_r, a_i)
        x_in_r = res_r + a_r
        x_in_i = res_i + a_i

        # --- 3. FFN (native layout) ---
        res_r, res_i = x_in_r, x_in_i
        f_r, f_i = self._run_ffn(self.ffn_in, x_in_r, x_in_i)
        f_r, f_i = self.norm3_in(f_r, f_i)
        x_in_r = res_r + f_r
        x_in_i = res_i + f_i

        # ============================================================
        # x_out stream
        # ============================================================

        # --- 1. Self-attention ---
        res_r, res_i = x_out_r, x_out_i
        q_r, q_i = self._to_attn(x_out_r, x_out_i)

        a_r, a_i, self.kv_cache_self_out = self.self_attn_out(
            q_r, q_i,
            q_r, q_i,
            q_r, q_i,
            pe_out_r, pe_out_i,
            pe_out_r, pe_out_i,
            kv_cache=self.kv_cache_self_out if use_cache else None,
            use_cache=use_cache,
        )

        a_r, a_i = self._from_attn(a_r, a_i)
        a_r, a_i = self.norm1_out(a_r, a_i)
        x_out_r = res_r + a_r
        x_out_i = res_i + a_i

        # --- 2. FFN (native layout) ---
        res_r, res_i = x_out_r, x_out_i
        f_r, f_i = self._run_ffn(self.ffn_out, x_out_r, x_out_i)
        f_r, f_i = self.norm2_out(f_r, f_i)
        x_out_r = res_r + f_r
        x_out_i = res_i + f_i

        return x_in_r, x_in_i, x_out_r, x_out_i

    def clear_cache(self):
        self.kv_cache_self_in  = None
        self.kv_cache_cross    = None
        self.kv_cache_self_out = None


# ===========================================================================
# __main__  —  tests
# ===========================================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- default geometry ----------
    B       = 4
    P_IN    = 34
    P_OUT   = 16
    BEAM    = 32
    FEA     = 96
    HEAD    = 8
    ATTN_DIM = BEAM * FEA // HEAD   # 384

    print("=" * 70)
    print(f"  Geometry:  beam={BEAM}  fea={FEA}  head={HEAD}  attn_dim={ATTN_DIM}")
    print(f"  Native:    x_in  [{B},{P_IN},{BEAM},{FEA}]   "
          f"x_out [{B},{P_OUT},{BEAM},{FEA}]")
    print(f"  Attn:      pe_in [{B},{HEAD},{P_IN},{ATTN_DIM}]   "
          f"pe_out [{B},{HEAD},{P_OUT},{ATTN_DIM}]")
    print("=" * 70)

    def make_inputs(p_in=P_IN, p_out=P_OUT):
        x_in_r  = torch.randn(B, p_in,  BEAM, FEA, device=device)
        x_in_i  = torch.randn(B, p_in,  BEAM, FEA, device=device)
        x_out_r = torch.randn(B, p_out, BEAM, FEA, device=device)
        x_out_i = torch.randn(B, p_out, BEAM, FEA, device=device)
        pe_in_r  = torch.randn(B, HEAD, p_in,  ATTN_DIM, device=device) * 0.1
        pe_in_i  = torch.randn(B, HEAD, p_in,  ATTN_DIM, device=device) * 0.1
        pe_out_r = torch.randn(B, HEAD, p_out, ATTN_DIM, device=device) * 0.1
        pe_out_i = torch.randn(B, HEAD, p_out, ATTN_DIM, device=device) * 0.1
        return (x_in_r, x_in_i, x_out_r, x_out_i,
                pe_in_r, pe_in_i, pe_out_r, pe_out_i)

    # ------------------------------------------------------------------
    # Test 1 — basic forward
    # ------------------------------------------------------------------
    print("\n[Test 1]  Basic forward  (no MoE, no cache)")
    model = TransformerBlock(
        beam=BEAM, fea=FEA, num_heads=HEAD,
        use_moe=False, use_kv_cache=False, depth=1
    ).to(device)

    inputs = make_inputs()
    out = model(*inputs)

    assert out[0].shape == (B, P_IN,  BEAM, FEA), f"x_in  bad: {out[0].shape}"
    assert out[2].shape == (B, P_OUT, BEAM, FEA), f"x_out bad: {out[2].shape}"
    print(f"  x_in:  {inputs[0].shape} → {out[0].shape}")
    print(f"  x_out: {inputs[2].shape} → {out[2].shape}")
    print(f"  params: {sum(p.numel() for p in model.parameters()):,}")
    print("  ✓ passed")

    # ------------------------------------------------------------------
    # Test 2 — MoE
    # ------------------------------------------------------------------
    print("\n[Test 2]  MoE  (8 experts, top-2)")
    model_moe = TransformerBlock(
        beam=BEAM, fea=FEA, num_heads=HEAD,
        use_moe=True, num_experts=8, top_k=2,
        use_kv_cache=False, depth=2
    ).to(device)

    out = model_moe(*inputs)
    assert out[0].shape == (B, P_IN,  BEAM, FEA)
    assert out[2].shape == (B, P_OUT, BEAM, FEA)
    print(f"  x_in:  {out[0].shape}   x_out: {out[2].shape}")
    print(f"  params: {sum(p.numel() for p in model_moe.parameters()):,}")
    print("  ✓ passed")

    # ------------------------------------------------------------------
    # Test 3 — KV cache incremental
    # ------------------------------------------------------------------
    print("\n[Test 3]  KV-cache incremental inference")
    model_cache = TransformerBlock(
        beam=BEAM, fea=FEA, num_heads=HEAD,
        use_moe=False, use_kv_cache=True, depth=1
    ).to(device)

    with torch.no_grad():
        _ = model_cache(*make_inputs(p_in=P_IN), use_cache=True)
    c1 = model_cache.kv_cache_self_in[0].shape[2]
    print(f"  step-1 cache p: {c1}  (expected {P_IN})")
    assert c1 == P_IN

    NEW = 5
    with torch.no_grad():
        _ = model_cache(*make_inputs(p_in=NEW), use_cache=True)
    c2 = model_cache.kv_cache_self_in[0].shape[2]
    print(f"  step-2 cache p: {c2}  (expected {P_IN + NEW})")
    assert c2 == P_IN + NEW

    model_cache.clear_cache()
    print("  ✓ passed")

    # ------------------------------------------------------------------
    # Test 4 — gradient flow
    # ------------------------------------------------------------------
    print("\n[Test 4]  Gradient back-prop  (MoE + DiffAttn)")
    model_grad = TransformerBlock(
        beam=BEAM, fea=FEA, num_heads=HEAD,
        use_moe=True, num_experts=8, top_k=2,
        use_kv_cache=False, depth=3
    ).to(device)

    out = model_grad(*make_inputs())
    loss = sum(o.sum() for o in out)
    loss.backward()

    n_grad  = sum(1 for p in model_grad.parameters() if p.grad is not None)
    n_total = sum(1 for p in model_grad.parameters())
    print(f"  params with grad: {n_grad}/{n_total}")
    assert n_grad == n_total, "gradient missing on some params"
    print("  ✓ passed")

    # ------------------------------------------------------------------
    # Test 5 — lambda_init schedule
    # ------------------------------------------------------------------
    print("\n[Test 5]  Lambda init values")
    print(f"  {'depth':>5} | {'lambda_init':>11}")
    print(f"  ------+------------")
    for d in [1, 2, 5, 10, 20]:
        print(f"  {d:>5} | {0.8 - 0.6 * math.exp(-0.3 * d):>11.4f}")

    # ------------------------------------------------------------------
    # Test 6 — param comparison
    # ------------------------------------------------------------------
    print("\n[Test 6]  Config comparison")
    print(f"  {'config':<28} | {'params':>12}")
    print(f"  ----------------------------+-------------")
    for name, kw in [
        ("no-MoE, no-cache",      dict(use_moe=False, use_kv_cache=False)),
        ("MoE-8/2, no-cache",     dict(use_moe=True,  num_experts=8,  top_k=2, use_kv_cache=False)),
        ("MoE-8/2  + cache",      dict(use_moe=True,  num_experts=8,  top_k=2, use_kv_cache=True)),
        ("MoE-16/4 + cache",      dict(use_moe=True,  num_experts=16, top_k=4, use_kv_cache=True)),
    ]:
        m = TransformerBlock(beam=BEAM, fea=FEA, num_heads=HEAD, depth=1, **kw).to(device)
        print(f"  {name:<28} | {sum(p.numel() for p in m.parameters()):>12,}")

    print("\n" + "=" * 70)
    print("  All tests passed ✓")
    print("=" * 70)
