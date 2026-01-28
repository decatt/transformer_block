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
        # x: [*, in_features]
        out_r = F.linear(x_r, self.weight_r) - F.linear(x_i, self.weight_i)
        out_i = F.linear(x_r, self.weight_i) + F.linear(x_i, self.weight_r)
        
        if self.bias_r is not None:
            out_r = out_r + self.bias_r
            out_i = out_i + self.bias_i
        
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
        
        # 复数的sigmoid: 对幅度应用sigmoid
        gate_mag = torch.sqrt(gate_r**2 + gate_i**2 + 1e-8)
        gate_phase = torch.atan2(gate_i, gate_r)
        gate_mag = torch.sigmoid(gate_mag)
        
        gate_r = gate_mag * torch.cos(gate_phase)
        gate_i = gate_mag * torch.sin(gate_phase)
        
        # 复数乘法
        out_r = gate_r * up_r - gate_i * up_i
        out_i = gate_r * up_i + gate_i * up_r
        
        return out_r, out_i


class ComplexLayerNorm(nn.Module):
    """对每个头分别进行复数归一化"""
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta_r = nn.Parameter(torch.zeros(normalized_shape))
        self.beta_i = nn.Parameter(torch.zeros(normalized_shape))
    
    def forward(self, x_r: torch.Tensor, x_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [b, head, seq, feature]
        # 对feature维度归一化
        mag = torch.sqrt(x_r**2 + x_i**2 + self.eps)
        mean_mag = mag.mean(dim=-1, keepdim=True)
        std_mag = mag.std(dim=-1, keepdim=True) + self.eps
        
        normalized_mag = (mag - mean_mag) / std_mag
        phase = torch.atan2(x_i, x_r)
        
        # 应用缩放
        normalized_mag = normalized_mag * self.gamma
        
        # 转回实部虚部
        out_r = normalized_mag * torch.cos(phase) + self.beta_r
        out_i = normalized_mag * torch.sin(phase) + self.beta_i
        
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
    """复数MoE层 - 只计算被选中的k个专家"""
    def __init__(self, dim: int, num_experts: int = 8, num_experts_per_token: int = 2, hidden_dim: int = None):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        hidden_dim = hidden_dim or dim * 4
        
        self.experts = nn.ModuleList([
            ComplexMoEExpert(dim, hidden_dim) for _ in range(num_experts)
        ])
        
        # Router: 使用复数幅度来计算路由分数
        self.router = ComplexLinear(dim, num_experts, bias=False)
    
    def forward(self, x_r: torch.Tensor, x_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_heads, seq_len, dim = x_r.shape
        
        # 计算路由分数
        router_logits_r, router_logits_i = self.router(x_r, x_i)
        router_logits = torch.sqrt(router_logits_r**2 + router_logits_i**2)  # [b, head, seq, num_experts]
        
        # 选择top-k专家
        routing_weights, selected_experts = torch.topk(router_logits, self.num_experts_per_token, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1)  # [b, head, seq, k]
        
        # 初始化输出
        out_r = torch.zeros_like(x_r)
        out_i = torch.zeros_like(x_i)
        
        # 只计算被选中的专家
        for i in range(self.num_experts_per_token):
            expert_indices = selected_experts[..., i]  # [b, head, seq]
            expert_weights = routing_weights[..., i:i+1]  # [b, head, seq, 1]
            
            for expert_id in range(self.num_experts):
                # 创建mask找到使用当前专家的token
                expert_mask = (expert_indices == expert_id)  # [b, head, seq]
                
                if expert_mask.any():
                    # 提取需要该专家处理的token
                    mask_expanded = expert_mask.unsqueeze(-1)  # [b, head, seq, 1]
                    
                    # 应用专家
                    expert_out_r, expert_out_i = self.experts[expert_id](x_r, x_i)
                    
                    # 加权并累加到输出（只对被选中的token）
                    weighted_out_r = expert_out_r * expert_weights * mask_expanded.float()
                    weighted_out_i = expert_out_i * expert_weights * mask_expanded.float()
                    
                    out_r = out_r + weighted_out_r
                    out_i = out_i + weighted_out_i
        
        return out_r, out_i


class ComplexAttention(nn.Module):
    """复数注意力机制（支持KV Cache）"""
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim
        self.scale = dim ** -0.5
        
        self.q_proj = ComplexLinear(dim, dim)
        self.k_proj = ComplexLinear(dim, dim)
        self.v_proj = ComplexLinear(dim, dim)
        self.gate_proj = ComplexLinear(dim, dim)  # 用于gate_score
        self.out_proj = ComplexLinear(dim, dim)
    
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
        # q: [b, head, seq_q, dim]
        # k, v: [b, head, seq_k, dim]
        
        # 投影
        q_proj_r, q_proj_i = self.q_proj(q_r, q_i)
        k_proj_r, k_proj_i = self.k_proj(k_r, k_i)
        v_proj_r, v_proj_i = self.v_proj(v_r, v_i)
        
        # 添加位置编码（复数加法）
        q_proj_r = q_proj_r + pe_q_r
        q_proj_i = q_proj_i + pe_q_i
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
        
        # 计算注意力分数（使用复数点积的幅度）
        # Q * K^H (共轭转置)
        scores_r = torch.matmul(q_proj_r, k_proj_r.transpose(-2, -1)) + \
                   torch.matmul(q_proj_i, k_proj_i.transpose(-2, -1))
        scores_i = torch.matmul(q_proj_i, k_proj_r.transpose(-2, -1)) - \
                   torch.matmul(q_proj_r, k_proj_i.transpose(-2, -1))
        
        scores = torch.sqrt(scores_r**2 + scores_i**2 + 1e-8) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        
        # 应用注意力权重
        attn_out_r = torch.matmul(attn_weights, v_proj_r)
        attn_out_i = torch.matmul(attn_weights, v_proj_i)
        
        # 计算gate score（对attention结果逐元素调制）
        gate_r, gate_i = self.gate_proj(q_r, q_i)
        
        # Gate调制（复数乘法）
        gated_r = gate_r * attn_out_r - gate_i * attn_out_i
        gated_i = gate_r * attn_out_i + gate_i * attn_out_r
        
        # 输出投影
        out_r, out_i = self.out_proj(gated_r, gated_i)
        
        return out_r, out_i, gate_r, gate_i, new_cache


class TransformerBlock(nn.Module):
    """
    复数Transformer Block with KV Cache and MoE
    
    Args:
        feature_dim: 特征维度
        num_heads: 注意力头数
        use_moe: 是否使用MoE
        num_experts: MoE专家数量
        num_experts_per_token: 每个token使用的专家数
        use_kv_cache: 是否使用KV Cache
    """
    def __init__(
        self,
        feature_dim: int,
        num_heads: int,
        use_moe: bool = True,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        use_kv_cache: bool = False
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.use_moe = use_moe
        self.use_kv_cache = use_kv_cache
        
        # Self-attention for x_in
        self.self_attn = ComplexAttention(feature_dim, num_heads)
        self.norm1_in = ComplexLayerNorm(feature_dim)
        
        # Cross-attention: x_in attend to x_out
        self.cross_attn = ComplexAttention(feature_dim, num_heads)
        self.norm2_in = ComplexLayerNorm(feature_dim)
        
        # FFN for x_in
        if use_moe:
            self.ffn_in = ComplexMoE(feature_dim, num_experts, num_experts_per_token)
        else:
            self.ffn_in = nn.Sequential()
            self.ffn_in.fc1 = ComplexLinear(feature_dim, feature_dim * 4)
            self.ffn_in.act = ComplexSwiGLU(feature_dim * 4)
            self.ffn_in.fc2 = ComplexLinear(feature_dim * 4, feature_dim)
        self.norm3_in = ComplexLayerNorm(feature_dim)
        
        # Self-attention for x_out
        self.self_attn_out = ComplexAttention(feature_dim, num_heads)
        self.norm1_out = ComplexLayerNorm(feature_dim)
        
        # FFN for x_out
        if use_moe:
            self.ffn_out = ComplexMoE(feature_dim, num_experts, num_experts_per_token)
        else:
            self.ffn_out = nn.Sequential()
            self.ffn_out.fc1 = ComplexLinear(feature_dim, feature_dim * 4)
            self.ffn_out.act = ComplexSwiGLU(feature_dim * 4)
            self.ffn_out.fc2 = ComplexLinear(feature_dim * 4, feature_dim)
        self.norm2_out = ComplexLayerNorm(feature_dim)
        
        # KV Cache存储
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
        """
        Args:
            x_in_r, x_in_i: [b, head, token_in_length, feature]
            x_out_r, x_out_i: [b, head, token_out_length, feature]
            pe_in_r, pe_in_i: [b, head, token_in_length, feature]
            pe_out_r, pe_out_i: [b, head, token_out_length, feature]
            use_cache: 是否使用缓存（None则使用类默认设置）
        
        Returns:
            x_in_r, x_in_i, x_out_r, x_out_i
        """
        if use_cache is None:
            use_cache = self.use_kv_cache
        
        # ============ Process x_in ============
        # 1. Self-attention on x_in
        residual_r, residual_i = x_in_r, x_in_i
        
        attn_out_r, attn_out_i, gate_r, gate_i, self.kv_cache_self_in = self.self_attn(
            x_in_r, x_in_i,
            x_in_r, x_in_i,
            x_in_r, x_in_i,
            pe_in_r, pe_in_i,
            pe_in_r, pe_in_i,
            kv_cache=self.kv_cache_self_in if use_cache else None,
            use_cache=use_cache
        )
        
        # Normalize and residual
        attn_out_r, attn_out_i = self.norm1_in(attn_out_r, attn_out_i)
        x_in_r = residual_r + attn_out_r
        x_in_i = residual_i + attn_out_i
        
        # 2. Cross-attention: x_in attend to x_out
        residual_r, residual_i = x_in_r, x_in_i
        
        cross_out_r, cross_out_i, _, _, self.kv_cache_cross = self.cross_attn(
            x_in_r, x_in_i,
            x_out_r, x_out_i,
            x_out_r, x_out_i,
            pe_in_r, pe_in_i,
            pe_out_r, pe_out_i,
            kv_cache=self.kv_cache_cross if use_cache else None,
            use_cache=use_cache
        )
        
        cross_out_r, cross_out_i = self.norm2_in(cross_out_r, cross_out_i)
        x_in_r = residual_r + cross_out_r
        x_in_i = residual_i + cross_out_i
        
        # 3. FFN for x_in
        residual_r, residual_i = x_in_r, x_in_i
        
        if self.use_moe:
            ffn_out_r, ffn_out_i = self.ffn_in(x_in_r, x_in_i)
        else:
            h_r, h_i = self.ffn_in.fc1(x_in_r, x_in_i)
            h_r, h_i = self.ffn_in.act(h_r, h_i)
            ffn_out_r, ffn_out_i = self.ffn_in.fc2(h_r, h_i)
        
        ffn_out_r, ffn_out_i = self.norm3_in(ffn_out_r, ffn_out_i)
        x_in_r = residual_r + ffn_out_r
        x_in_i = residual_i + ffn_out_i
        
        # ============ Process x_out ============
        # 1. Self-attention on x_out
        residual_r, residual_i = x_out_r, x_out_i
        
        attn_out_r, attn_out_i, _, _, self.kv_cache_self_out = self.self_attn_out(
            x_out_r, x_out_i,
            x_out_r, x_out_i,
            x_out_r, x_out_i,
            pe_out_r, pe_out_i,
            pe_out_r, pe_out_i,
            kv_cache=self.kv_cache_self_out if use_cache else None,
            use_cache=use_cache
        )
        
        attn_out_r, attn_out_i = self.norm1_out(attn_out_r, attn_out_i)
        x_out_r = residual_r + attn_out_r
        x_out_i = residual_i + attn_out_i
        
        # 2. FFN for x_out
        residual_r, residual_i = x_out_r, x_out_i
        
        if self.use_moe:
            ffn_out_r, ffn_out_i = self.ffn_out(x_out_r, x_out_i)
        else:
            h_r, h_i = self.ffn_out.fc1(x_out_r, x_out_i)
            h_r, h_i = self.ffn_out.act(h_r, h_i)
            ffn_out_r, ffn_out_i = self.ffn_out.fc2(h_r, h_i)
        
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
    print("复数Transformer Block测试")
    print("=" * 80)
    
    # 测试参数
    batch_size = 4
    num_heads = 8
    token_in_length = 34
    token_out_length = 20
    feature_dim = 384
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    # 创建模型
    print("\n" + "=" * 80)
    print("测试1: 基础功能测试（无MoE，无KV Cache）")
    print("=" * 80)
    model_basic = TransformerBlock(
        feature_dim=feature_dim,
        num_heads=num_heads,
        use_moe=False,
        use_kv_cache=False
    ).to(device)
    
    # 创建输入数据
    x_in_r = torch.randn(batch_size, num_heads, token_in_length, feature_dim).to(device)
    x_in_i = torch.randn(batch_size, num_heads, token_in_length, feature_dim).to(device)
    x_out_r = torch.randn(batch_size, num_heads, token_out_length, feature_dim).to(device)
    x_out_i = torch.randn(batch_size, num_heads, token_out_length, feature_dim).to(device)
    
    pe_in_r = torch.randn(batch_size, num_heads, token_in_length, feature_dim).to(device) * 0.1
    pe_in_i = torch.randn(batch_size, num_heads, token_in_length, feature_dim).to(device) * 0.1
    pe_out_r = torch.randn(batch_size, num_heads, token_out_length, feature_dim).to(device) * 0.1
    pe_out_i = torch.randn(batch_size, num_heads, token_out_length, feature_dim).to(device) * 0.1
    
    # 前向传播
    out_in_r, out_in_i, out_out_r, out_out_i = model_basic(
        x_in_r, x_in_i, x_out_r, x_out_i,
        pe_in_r, pe_in_i, pe_out_r, pe_out_i
    )
    
    print(f"输入 x_in shape: {x_in_r.shape}")
    print(f"输入 x_out shape: {x_out_r.shape}")
    print(f"输出 x_in shape: {out_in_r.shape}")
    print(f"输出 x_out shape: {out_out_r.shape}")
    print(f"参数数量: {sum(p.numel() for p in model_basic.parameters()):,}")
    
    # 测试MoE
    print("\n" + "=" * 80)
    print("测试2: MoE功能测试")
    print("=" * 80)
    model_moe = TransformerBlock(
        feature_dim=feature_dim,
        num_heads=num_heads,
        use_moe=True,
        num_experts=8,
        num_experts_per_token=2,
        use_kv_cache=False
    ).to(device)
    
    out_in_r, out_in_i, out_out_r, out_out_i = model_moe(
        x_in_r, x_in_i, x_out_r, x_out_i,
        pe_in_r, pe_in_i, pe_out_r, pe_out_i
    )
    
    print(f"MoE输出 x_in shape: {out_in_r.shape}")
    print(f"MoE输出 x_out shape: {out_out_r.shape}")
    print(f"MoE参数数量: {sum(p.numel() for p in model_moe.parameters()):,}")
    
    # 测试KV Cache
    print("\n" + "=" * 80)
    print("测试3: KV Cache功能测试")
    print("=" * 80)
    model_cache = TransformerBlock(
        feature_dim=feature_dim,
        num_heads=num_heads,
        use_moe=False,
        use_kv_cache=True
    ).to(device)
    
    # 第一次前向传播（建立cache）
    print("\n第一次前向传播（建立cache）...")
    out_in_r_1, out_in_i_1, out_out_r_1, out_out_i_1 = model_cache(
        x_in_r, x_in_i, x_out_r, x_out_i,
        pe_in_r, pe_in_i, pe_out_r, pe_out_i,
        use_cache=True
    )
    
    # 检查cache
    print(f"Self-attention in cache shape: {model_cache.kv_cache_self_in[0].shape if model_cache.kv_cache_self_in else 'None'}")
    print(f"Cross-attention cache shape: {model_cache.kv_cache_cross[0].shape if model_cache.kv_cache_cross else 'None'}")
    
    # 第二次前向传播（使用cache，输入更短序列）
    print("\n第二次前向传播（使用cache）...")
    x_in_r_short = x_in_r[:, :, :5, :]  # 只用前5个token
    x_in_i_short = x_in_i[:, :, :5, :]
    pe_in_r_short = pe_in_r[:, :, :5, :]
    pe_in_i_short = pe_in_i[:, :, :5, :]
    
    out_in_r_2, out_in_i_2, out_out_r_2, out_out_i_2 = model_cache(
        x_in_r_short, x_in_i_short, x_out_r, x_out_i,
        pe_in_r_short, pe_in_i_short, pe_out_r, pe_out_i,
        use_cache=True
    )
    
    print(f"更新后的cache shape: {model_cache.kv_cache_self_in[0].shape}")
    
    # 清除cache
    model_cache.clear_cache()
    print("Cache已清除")
    
    # 梯度测试
    print("\n" + "=" * 80)
    print("测试4: 梯度反向传播测试")
    print("=" * 80)
    
    # 计算损失
    loss = (out_in_r.sum() + out_in_i.sum() + out_out_r.sum() + out_out_i.sum())
    loss.backward()
    
    # 检查梯度
    has_grad = sum(1 for p in model_basic.parameters() if p.grad is not None)
    total_params = sum(1 for p in model_basic.parameters())
    print(f"有梯度的参数: {has_grad}/{total_params}")
    print(f"梯度正常: {'✓' if has_grad == total_params else '✗'}")
    
    # 复数性质测试
    print("\n" + "=" * 80)
    print("测试5: 复数性质验证")
    print("=" * 80)
    
    # 测试输入的幅度和相位是否被保留在合理范围内
    input_mag = torch.sqrt(x_in_r**2 + x_in_i**2).mean()
    output_mag = torch.sqrt(out_in_r**2 + out_in_i**2).mean()
    
    print(f"输入平均幅度: {input_mag.item():.4f}")
    print(f"输出平均幅度: {output_mag.item():.4f}")
    print(f"幅度比: {(output_mag/input_mag).item():.4f}")
    
    print("\n" + "=" * 80)
    print("所有测试完成！")
    print("=" * 80)
