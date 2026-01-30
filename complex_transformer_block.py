import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class ComplexLinear(nn.Module):
    """复数线性层: (x_r, x_i) = (w_r*x_r - w_i*x_i, w_r*x_i + w_i*x_r)"""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight_r = nn.Parameter(torch.randn(out_features, in_features) / math.sqrt(in_features))
        self.weight_i = nn.Parameter(torch.randn(out_features, in_features) / math.sqrt(in_features))
        
        if bias:
            self.bias_r = nn.Parameter(torch.zeros(out_features))
            self.bias_i = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias_r', None)
            self.register_parameter('bias_i', None)
    
    def forward(self, x_r: torch.Tensor, x_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out_r = F.linear(x_r, self.weight_r) - F.linear(x_i, self.weight_i)
        out_i = F.linear(x_r, self.weight_i) + F.linear(x_i, self.weight_r)
        
        if self.bias_r is not None:
            out_r = out_r + self.bias_r
            out_i = out_i + self.bias_i
        
        return out_r, out_i


class ComplexRMSNorm(nn.Module):
    """复数RMSNorm - 更简单高效的归一化"""
    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)
    
    def forward(self, x_r: torch.Tensor, x_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 计算复数幅度的RMS
        mag_squared = x_r ** 2 + x_i ** 2
        rms = torch.sqrt(mag_squared.mean(dim=-1, keepdim=True) + self.eps)
        
        # 归一化
        out_r = x_r / rms
        out_i = x_i / rms
        
        # 应用可学习的缩放
        if self.elementwise_affine:
            out_r = out_r * self.weight
            out_i = out_i * self.weight
        
        return out_r, out_i


class ComplexSwiGLU(nn.Module):
    """复数SwiGLU激活函数"""
    def __init__(self, dim: int):
        super().__init__()
        self.gate_proj = ComplexLinear(dim, dim, bias=False)
        self.up_proj = ComplexLinear(dim, dim, bias=False)
    
    def forward(self, x_r: torch.Tensor, x_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gate_r, gate_i = self.gate_proj(x_r, x_i)
        up_r, up_i = self.up_proj(x_r, x_i)
        
        # 对幅度应用SiLU
        gate_mag = torch.sqrt(gate_r**2 + gate_i**2 + 1e-8)
        gate_phase = torch.atan2(gate_i, gate_r)
        gate_mag = F.silu(gate_mag)
        
        gate_r = gate_mag * torch.cos(gate_phase)
        gate_i = gate_mag * torch.sin(gate_phase)
        
        # 复数乘法
        out_r = gate_r * up_r - gate_i * up_i
        out_i = gate_r * up_i + gate_i * up_r
        
        return out_r, out_i


class ComplexMoEExpert(nn.Module):
    """单个MoE专家"""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = ComplexLinear(dim, hidden_dim)
        self.fc2 = ComplexLinear(hidden_dim, dim)
        self.activation = ComplexSwiGLU(hidden_dim)
    
    def forward(self, x_r: torch.Tensor, x_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_r, h_i = self.fc1(x_r, x_i)
        h_r, h_i = self.activation(h_r, h_i)
        out_r, out_i = self.fc2(h_r, h_i)
        return out_r, out_i


class ComplexMoE(nn.Module):
    """复数MoE层"""
    def __init__(self, dim: int, num_experts: int = 8, num_experts_per_token: int = 2, hidden_dim: int = None):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        hidden_dim = hidden_dim or dim * 4
        
        self.experts = nn.ModuleList([
            ComplexMoEExpert(dim, hidden_dim) for _ in range(num_experts)
        ])
        
        self.router = ComplexLinear(dim, num_experts, bias=False)
    
    def forward(self, x_r: torch.Tensor, x_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_heads, seq_len, dim = x_r.shape
        
        router_logits_r, router_logits_i = self.router(x_r, x_i)
        router_logits = torch.sqrt(router_logits_r**2 + router_logits_i**2)
        
        routing_weights, selected_experts = torch.topk(router_logits, self.num_experts_per_token, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        out_r = torch.zeros_like(x_r)
        out_i = torch.zeros_like(x_i)
        
        for i in range(self.num_experts_per_token):
            expert_indices = selected_experts[..., i]
            expert_weights = routing_weights[..., i:i+1]
            
            for expert_id in range(self.num_experts):
                expert_mask = (expert_indices == expert_id)
                
                if expert_mask.any():
                    mask_expanded = expert_mask.unsqueeze(-1)
                    expert_out_r, expert_out_i = self.experts[expert_id](x_r, x_i)
                    
                    weighted_out_r = expert_out_r * expert_weights * mask_expanded.float()
                    weighted_out_i = expert_out_i * expert_weights * mask_expanded.float()
                    
                    out_r = out_r + weighted_out_r
                    out_i = out_i + weighted_out_i
        
        return out_r, out_i


class ComplexDifferentialAttention(nn.Module):
    """
    复数Differential Attention - 基于差分注意力机制
    
    核心思想：使用两组注意力头的差值来消除噪声，增强信号
    - 每个逻辑头实际由2个物理头组成
    - attn_out = attn1 - lambda * attn2
    - lambda通过可学习参数动态调整
    """
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert num_heads % 2 == 0, "num_heads必须是偶数以支持differential attention"
        
        self.num_heads = num_heads  # 逻辑头数
        self.num_physical_heads = num_heads * 2  # 物理头数（每个逻辑头=2个物理头）
        self.head_dim = dim
        self.scale = dim ** -0.5
        
        # Q投影到2倍头数
        self.q_proj = ComplexLinear(dim, dim * 2)  # 输出2h
        self.k_proj = ComplexLinear(dim, dim)
        self.v_proj = ComplexLinear(dim, dim)
        self.gate_proj = ComplexLinear(dim, dim)
        self.out_proj = ComplexLinear(dim, dim)
        
        # Lambda参数 - 每个逻辑头有独立的lambda
        # 参考图片中的设置：4个lambda参数在所有头之间共享
        self.lambda_q1 = nn.Parameter(torch.zeros(num_heads))
        self.lambda_k1 = nn.Parameter(torch.zeros(num_heads))
        self.lambda_q2 = nn.Parameter(torch.zeros(num_heads))
        self.lambda_k2 = nn.Parameter(torch.zeros(num_heads))
        
        # Lambda初始化函数（参考图片）
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * 1)  # depth=1作为默认
        
        # SubLN (RMSNorm for differential attention)
        self.subln = ComplexRMSNorm(2 * dim, elementwise_affine=True)
    
    def lambda_init_fn(self, depth: int) -> float:
        """推荐的lambda初始化值"""
        return 0.8 - 0.6 * math.exp(-0.3 * depth)
    
    def forward(
        self,
        q_r: torch.Tensor, q_i: torch.Tensor,
        k_r: torch.Tensor, k_i: torch.Tensor,
        v_r: torch.Tensor, v_i: torch.Tensor,
        pe_q_r: torch.Tensor, pe_q_i: torch.Tensor,
        pe_k_r: torch.Tensor, pe_k_i: torch.Tensor,
        kv_cache: Optional[Tuple] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple]]:
        b, h, seq_q, d = q_r.shape
        _, _, seq_k, _ = k_r.shape
        
        # Q投影（2倍头数）
        q_proj_r, q_proj_i = self.q_proj(q_r, q_i)  # [b, h, seq_q, 2*d]
        k_proj_r, k_proj_i = self.k_proj(k_r, k_i)  # [b, h, seq_k, d]
        v_proj_r, v_proj_i = self.v_proj(v_r, v_i)  # [b, h, seq_k, d]
        
        # 添加位置编码
        pe_q_r_expanded = torch.cat([pe_q_r, pe_q_r], dim=-1)  # 扩展到2倍
        pe_q_i_expanded = torch.cat([pe_q_i, pe_q_i], dim=-1)
        q_proj_r = q_proj_r + pe_q_r_expanded
        q_proj_i = q_proj_i + pe_q_i_expanded
        k_proj_r = k_proj_r + pe_k_r
        k_proj_i = k_proj_i + pe_k_i
        
        # KV Cache处理
        new_cache = None
        if use_cache:
            if kv_cache is not None:
                cached_k_r, cached_k_i, cached_v_r, cached_v_i = kv_cache
                k_proj_r = torch.cat([cached_k_r, k_proj_r], dim=2)
                k_proj_i = torch.cat([cached_k_i, k_proj_i], dim=2)
                v_proj_r = torch.cat([cached_v_r, v_proj_r], dim=2)
                v_proj_i = torch.cat([cached_v_i, v_proj_i], dim=2)
            new_cache = (k_proj_r, k_proj_i, v_proj_r, v_proj_i)
        
        seq_k = k_proj_r.shape[2]  # 更新序列长度（可能包含cache）
        
        # 分离成两组头（attn1和attn2）
        # q: [b, h, seq_q, 2*d] -> [b, h, seq_q, d, 2]
        q_proj_r = q_proj_r.view(b, h, seq_q, d, 2)
        q_proj_i = q_proj_i.view(b, h, seq_q, d, 2)
        
        q1_r, q2_r = q_proj_r[..., 0], q_proj_r[..., 1]  # [b, h, seq_q, d]
        q1_i, q2_i = q_proj_i[..., 0], q_proj_i[..., 1]
        
        # 计算两组注意力分数
        # Attention 1
        scores1_r = torch.matmul(q1_r, k_proj_r.transpose(-2, -1)) + \
                    torch.matmul(q1_i, k_proj_i.transpose(-2, -1))
        scores1_i = torch.matmul(q1_i, k_proj_r.transpose(-2, -1)) - \
                    torch.matmul(q1_r, k_proj_i.transpose(-2, -1))
        scores1 = torch.sqrt(scores1_r**2 + scores1_i**2 + 1e-8) * self.scale
        attn_weights1 = F.softmax(scores1, dim=-1)
        
        # Attention 2
        scores2_r = torch.matmul(q2_r, k_proj_r.transpose(-2, -1)) + \
                    torch.matmul(q2_i, k_proj_i.transpose(-2, -1))
        scores2_i = torch.matmul(q2_i, k_proj_r.transpose(-2, -1)) - \
                    torch.matmul(q2_r, k_proj_i.transpose(-2, -1))
        scores2 = torch.sqrt(scores2_r**2 + scores2_i**2 + 1e-8) * self.scale
        attn_weights2 = F.softmax(scores2, dim=-1)
        
        # 应用注意力权重到V
        attn_out1_r = torch.matmul(attn_weights1, v_proj_r)
        attn_out1_i = torch.matmul(attn_weights1, v_proj_i)
        
        attn_out2_r = torch.matmul(attn_weights2, v_proj_r)
        attn_out2_i = torch.matmul(attn_weights2, v_proj_i)
        
        # 计算lambda值 (re-parameterization)
        # lambda = exp(sum(lambda_q * lambda_k))
        lambda_q1 = self.lambda_q1.view(1, h, 1, 1)  # [1, h, 1, 1]
        lambda_k1 = self.lambda_k1.view(1, h, 1, 1)
        lambda_q2 = self.lambda_q2.view(1, h, 1, 1)
        lambda_k2 = self.lambda_k2.view(1, h, 1, 1)
        
        lambda_1 = torch.exp(lambda_q1 * lambda_k1)
        lambda_2 = torch.exp(lambda_q2 * lambda_k2)
        
        # lambda_full = lambda_1 - lambda_2 + lambda_init
        lambda_val = lambda_1 - lambda_2 + self.lambda_init
        lambda_val = torch.sigmoid(lambda_val)  # 归一化到[0,1]
        
        # Differential Attention: attn = attn1 - lambda * attn2
        attn_out_r = attn_out1_r - lambda_val * attn_out2_r
        attn_out_i = attn_out1_i - lambda_val * attn_out2_i
        
        # SubLN (GroupNorm style normalization)
        # 将attn1和attn2拼接后归一化
        concat_r = torch.stack([attn_out1_r, attn_out2_r], dim=-1)  # [b, h, seq_q, d, 2]
        concat_i = torch.stack([attn_out1_i, attn_out2_i], dim=-1)
        concat_r = concat_r.reshape(b, h, seq_q, 2*d)
        concat_i = concat_i.reshape(b, h, seq_q, 2*d)
        
        normed_r, normed_i = self.subln(concat_r, concat_i)
        
        # 提取差分注意力部分（前d维对应attn_out）
        attn_out_r = normed_r[..., :d]
        attn_out_i = normed_i[..., :d]
        
        # Gate调制
        gate_r, gate_i = self.gate_proj(q_r, q_i)
        gated_r = gate_r * attn_out_r - gate_i * attn_out_i
        gated_i = gate_r * attn_out_i + gate_i * attn_out_r
        
        # 输出投影
        out_r, out_i = self.out_proj(gated_r, gated_i)
        
        return out_r, out_i, gate_r, gate_i, new_cache


class TransformerBlock(nn.Module):
    """
    复数Transformer Block with Differential Attention, KV Cache and MoE
    
    Args:
        feature_dim: 特征维度
        num_heads: 注意力头数（必须是偶数用于differential attention）
        use_moe: 是否使用MoE
        num_experts: MoE专家数量
        num_experts_per_token: 每个token使用的专家数
        use_kv_cache: 是否使用KV Cache
        use_diff_attn: 是否使用Differential Attention
        depth: 层深度（用于lambda初始化）
    """
    def __init__(
        self,
        feature_dim: int,
        num_heads: int,
        use_moe: bool = True,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        use_kv_cache: bool = False,
        use_diff_attn: bool = True,
        depth: int = 1
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.use_moe = use_moe
        self.use_kv_cache = use_kv_cache
        self.use_diff_attn = use_diff_attn
        
        # 选择注意力类型
        if use_diff_attn:
            assert num_heads % 2 == 0, "使用Differential Attention时，num_heads必须是偶数"
            AttentionClass = ComplexDifferentialAttention
        else:
            # 这里需要之前的ComplexAttention类（未在此文件中定义，需要从原文件导入）
            raise NotImplementedError("请使用Differential Attention或导入原始ComplexAttention")
        
        # Self-attention for x_in
        self.self_attn = AttentionClass(feature_dim, num_heads)
        if hasattr(self.self_attn, 'lambda_init_fn'):
            # 设置深度相关的lambda初始化
            self.self_attn.lambda_init = self.self_attn.lambda_init_fn(depth)
        self.norm1_in = ComplexRMSNorm(feature_dim)
        
        # Cross-attention
        self.cross_attn = AttentionClass(feature_dim, num_heads)
        if hasattr(self.cross_attn, 'lambda_init_fn'):
            self.cross_attn.lambda_init = self.cross_attn.lambda_init_fn(depth)
        self.norm2_in = ComplexRMSNorm(feature_dim)
        
        # FFN for x_in
        if use_moe:
            self.ffn_in = ComplexMoE(feature_dim, num_experts, num_experts_per_token)
        else:
            self.ffn_in = nn.ModuleDict({
                'fc1': ComplexLinear(feature_dim, feature_dim * 4),
                'act': ComplexSwiGLU(feature_dim * 4),
                'fc2': ComplexLinear(feature_dim * 4, feature_dim)
            })
        self.norm3_in = ComplexRMSNorm(feature_dim)
        
        # Self-attention for x_out
        self.self_attn_out = AttentionClass(feature_dim, num_heads)
        if hasattr(self.self_attn_out, 'lambda_init_fn'):
            self.self_attn_out.lambda_init = self.self_attn_out.lambda_init_fn(depth)
        self.norm1_out = ComplexRMSNorm(feature_dim)
        
        # FFN for x_out
        if use_moe:
            self.ffn_out = ComplexMoE(feature_dim, num_experts, num_experts_per_token)
        else:
            self.ffn_out = nn.ModuleDict({
                'fc1': ComplexLinear(feature_dim, feature_dim * 4),
                'act': ComplexSwiGLU(feature_dim * 4),
                'fc2': ComplexLinear(feature_dim * 4, feature_dim)
            })
        self.norm2_out = ComplexRMSNorm(feature_dim)
        
        # KV Cache
        self.kv_cache_self_in = None
        self.kv_cache_cross = None
        self.kv_cache_self_out = None
    
    def forward(
        self,
        x_in_r: torch.Tensor,
        x_in_i: torch.Tensor,
        x_out_r: torch.Tensor,
        x_out_i: torch.Tensor,
        pe_in_r: torch.Tensor,
        pe_in_i: torch.Tensor,
        pe_out_r: torch.Tensor,
        pe_out_i: torch.Tensor,
        use_cache: bool = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if use_cache is None:
            use_cache = self.use_kv_cache
        
        # Process x_in
        residual_r, residual_i = x_in_r, x_in_i
        
        attn_out_r, attn_out_i, _, _, self.kv_cache_self_in = self.self_attn(
            x_in_r, x_in_i, x_in_r, x_in_i, x_in_r, x_in_i,
            pe_in_r, pe_in_i, pe_in_r, pe_in_i,
            kv_cache=self.kv_cache_self_in if use_cache else None,
            use_cache=use_cache
        )
        
        attn_out_r, attn_out_i = self.norm1_in(attn_out_r, attn_out_i)
        x_in_r = residual_r + attn_out_r
        x_in_i = residual_i + attn_out_i
        
        # Cross-attention
        residual_r, residual_i = x_in_r, x_in_i
        
        cross_out_r, cross_out_i, _, _, self.kv_cache_cross = self.cross_attn(
            x_in_r, x_in_i, x_out_r, x_out_i, x_out_r, x_out_i,
            pe_in_r, pe_in_i, pe_out_r, pe_out_i,
            kv_cache=self.kv_cache_cross if use_cache else None,
            use_cache=use_cache
        )
        
        cross_out_r, cross_out_i = self.norm2_in(cross_out_r, cross_out_i)
        x_in_r = residual_r + cross_out_r
        x_in_i = residual_i + cross_out_i
        
        # FFN for x_in
        residual_r, residual_i = x_in_r, x_in_i
        
        if self.use_moe:
            ffn_out_r, ffn_out_i = self.ffn_in(x_in_r, x_in_i)
        else:
            h_r, h_i = self.ffn_in['fc1'](x_in_r, x_in_i)
            h_r, h_i = self.ffn_in['act'](h_r, h_i)
            ffn_out_r, ffn_out_i = self.ffn_in['fc2'](h_r, h_i)
        
        ffn_out_r, ffn_out_i = self.norm3_in(ffn_out_r, ffn_out_i)
        x_in_r = residual_r + ffn_out_r
        x_in_i = residual_i + ffn_out_i
        
        # Process x_out
        residual_r, residual_i = x_out_r, x_out_i
        
        attn_out_r, attn_out_i, _, _, self.kv_cache_self_out = self.self_attn_out(
            x_out_r, x_out_i, x_out_r, x_out_i, x_out_r, x_out_i,
            pe_out_r, pe_out_i, pe_out_r, pe_out_i,
            kv_cache=self.kv_cache_self_out if use_cache else None,
            use_cache=use_cache
        )
        
        attn_out_r, attn_out_i = self.norm1_out(attn_out_r, attn_out_i)
        x_out_r = residual_r + attn_out_r
        x_out_i = residual_i + attn_out_i
        
        # FFN for x_out
        residual_r, residual_i = x_out_r, x_out_i
        
        if self.use_moe:
            ffn_out_r, ffn_out_i = self.ffn_out(x_out_r, x_out_i)
        else:
            h_r, h_i = self.ffn_out['fc1'](x_out_r, x_out_i)
            h_r, h_i = self.ffn_out['act'](h_r, h_i)
            ffn_out_r, ffn_out_i = self.ffn_out['fc2'](h_r, h_i)
        
        ffn_out_r, ffn_out_i = self.norm2_out(ffn_out_r, ffn_out_i)
        x_out_r = residual_r + ffn_out_r
        x_out_i = residual_i + ffn_out_i
        
        return x_in_r, x_in_i, x_out_r, x_out_i
    
    def clear_cache(self):
        """清除KV Cache"""
        self.kv_cache_self_in = None
        self.kv_cache_cross = None
        self.kv_cache_self_out = None


if __name__ == "__main__":
    print("=" * 80)
    print("复数Transformer Block with Differential Attention 测试")
    print("=" * 80)
    
    # 测试参数
    batch_size = 4
    num_heads = 8  # 必须是偶数
    token_in_length = 34
    token_out_length = 20
    feature_dim = 384
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    # 测试1: Differential Attention
    print("\n" + "=" * 80)
    print("测试1: Differential Attention基础功能")
    print("=" * 80)
    
    model_diff = TransformerBlock(
        feature_dim=feature_dim,
        num_heads=num_heads,
        use_moe=False,
        use_kv_cache=False,
        use_diff_attn=True,
        depth=1
    ).to(device)
    
    # 创建输入
    x_in_r = torch.randn(batch_size, num_heads, token_in_length, feature_dim).to(device)
    x_in_i = torch.randn(batch_size, num_heads, token_in_length, feature_dim).to(device)
    x_out_r = torch.randn(batch_size, num_heads, token_out_length, feature_dim).to(device)
    x_out_i = torch.randn(batch_size, num_heads, token_out_length, feature_dim).to(device)
    
    pe_in_r = torch.randn(batch_size, num_heads, token_in_length, feature_dim).to(device) * 0.1
    pe_in_i = torch.randn(batch_size, num_heads, token_in_length, feature_dim).to(device) * 0.1
    pe_out_r = torch.randn(batch_size, num_heads, token_out_length, feature_dim).to(device) * 0.1
    pe_out_i = torch.randn(batch_size, num_heads, token_out_length, feature_dim).to(device) * 0.1
    
    out_in_r, out_in_i, out_out_r, out_out_i = model_diff(
        x_in_r, x_in_i, x_out_r, x_out_i,
        pe_in_r, pe_in_i, pe_out_r, pe_out_i
    )
    
    print(f"输出shape: {out_in_r.shape}")
    print(f"参数数量: {sum(p.numel() for p in model_diff.parameters()):,}")
    
    # 检查lambda参数
    print("\nLambda参数:")
    print(f"  lambda_q1: {model_diff.self_attn.lambda_q1.data}")
    print(f"  lambda_init: {model_diff.self_attn.lambda_init:.4f}")
    
    # 测试2: Differential Attention + MoE
    print("\n" + "=" * 80)
    print("测试2: Differential Attention + MoE")
    print("=" * 80)
    
    model_diff_moe = TransformerBlock(
        feature_dim=feature_dim,
        num_heads=num_heads,
        use_moe=True,
        num_experts=8,
        num_experts_per_token=2,
        use_kv_cache=False,
        use_diff_attn=True,
        depth=2
    ).to(device)
    
    out_in_r, out_in_i, out_out_r, out_out_i = model_diff_moe(
        x_in_r, x_in_i, x_out_r, x_out_i,
        pe_in_r, pe_in_i, pe_out_r, pe_out_i
    )
    
    print(f"MoE输出shape: {out_in_r.shape}")
    print(f"MoE参数数量: {sum(p.numel() for p in model_diff_moe.parameters()):,}")
    
    # 测试3: 全功能（Differential Attention + MoE + KV Cache）
    print("\n" + "=" * 80)
    print("测试3: Differential Attention + MoE + KV Cache")
    print("=" * 80)
    
    model_full = TransformerBlock(
        feature_dim=feature_dim,
        num_heads=num_heads,
        use_moe=True,
        num_experts=8,
        num_experts_per_token=2,
        use_kv_cache=True,
        use_diff_attn=True,
        depth=3
    ).to(device)
    
    # 第一次前向传播
    print("第一次前向传播（建立cache）...")
    out1 = model_full(
        x_in_r, x_in_i, x_out_r, x_out_i,
        pe_in_r, pe_in_i, pe_out_r, pe_out_i,
        use_cache=True
    )
    
    print(f"Cache shape: {model_full.kv_cache_self_in[0].shape if model_full.kv_cache_self_in else 'None'}")
    
    # 第二次前向传播（使用cache）
    print("第二次前向传播（使用cache）...")
    x_in_r_short = x_in_r[:, :, :5, :]
    x_in_i_short = x_in_i[:, :, :5, :]
    pe_in_r_short = pe_in_r[:, :, :5, :]
    pe_in_i_short = pe_in_i[:, :, :5, :]
    
    out2 = model_full(
        x_in_r_short, x_in_i_short, x_out_r, x_out_i,
        pe_in_r_short, pe_in_i_short, pe_out_r, pe_out_i,
        use_cache=True
    )
    
    print(f"更新后cache shape: {model_full.kv_cache_self_in[0].shape}")
    
    # 测试4: 梯度测试
    print("\n" + "=" * 80)
    print("测试4: 梯度反向传播")
    print("=" * 80)
    
    model_full.clear_cache()
    out_in_r, out_in_i, out_out_r, out_out_i = model_full(
        x_in_r, x_in_i, x_out_r, x_out_i,
        pe_in_r, pe_in_i, pe_out_r, pe_out_i,
        use_cache=False
    )
    
    loss = (out_in_r.sum() + out_in_i.sum() + out_out_r.sum() + out_out_i.sum())
    loss.backward()
    
    has_grad = sum(1 for p in model_full.parameters() if p.grad is not None)
    total_params = sum(1 for p in model_full.parameters())
    print(f"有梯度的参数: {has_grad}/{total_params}")
    print(f"梯度正常: {'✓' if has_grad == total_params else '✗'}")
    
    # 测试5: Lambda值分析
    print("\n" + "=" * 80)
    print("测试5: Lambda参数分析")
    print("=" * 80)
    
    for depth in [1, 5, 10, 20]:
        lambda_init = 0.8 - 0.6 * math.exp(-0.3 * depth)
        print(f"Depth {depth:2d}: lambda_init = {lambda_init:.4f}")
    
    print("\n" + "=" * 80)
    print("所有测试完成！")
    print("=" * 80)
    
    # 性能对比总结
    print("\n性能对比:")
    print(f"  基础模型参数: ~{sum(p.numel() for p in model_diff.parameters()):,}")
    print(f"  MoE模型参数: ~{sum(p.numel() for p in model_diff_moe.parameters()):,}")
    print(f"  完整模型参数: ~{sum(p.numel() for p in model_full.parameters()):,}")
