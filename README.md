# 复数Transformer Block实现说明

## 代码结构概览

我已经为您实现了一个完整的复数Transformer Block，包含以下关键功能：

### 1. 核心组件

#### ComplexLinear (复数线性层)
```python
复数线性变换: 
out_r = w_r * x_r - w_i * x_i
out_i = w_r * x_i + w_i * x_r
```

#### ComplexLayerNorm (复数归一化)
- 对每个头的特征维度进行归一化
- 先计算复数幅度，归一化后保持相位信息

#### ComplexSwiGLU (复数激活函数)
- 对复数幅度应用sigmoid
- 保持相位信息进行门控

### 2. 高级功能

#### KV Cache (键值缓存)
```python
特性：
- 支持增量推理
- 自动累积历史K/V
- 可随时清除缓存
```

#### MoE (混合专家模型)
```python
特性：
- 只计算被选中的k个专家
- 基于复数幅度的路由机制
- Top-k选择 + Softmax加权
```

### 3. 注意力机制

#### ComplexAttention
```python
功能：
1. Q, K, V投影（复数线性变换）
2. 添加位置编码
3. 计算注意力分数（使用复数共轭点积）
4. Gate Score调制（逐元素复数乘法）
5. 支持KV Cache
```

### 4. TransformerBlock结构

```
输入: x_in, x_out (各包含实部和虚部)

对 x_in:
  1. Self-Attention + Norm + Residual
  2. Cross-Attention (attend to x_out) + Norm + Residual
  3. FFN/MoE + Norm + Residual

对 x_out:
  1. Self-Attention + Norm + Residual
  2. FFN/MoE + Norm + Residual

输出: 处理后的 x_in, x_out
```

## 使用示例

### 基础使用
```python
model = TransformerBlock(
    feature_dim=384,
    num_heads=8,
    use_moe=False,
    use_kv_cache=False
)

# 前向传播
out_in_r, out_in_i, out_out_r, out_out_i = model(
    x_in_r, x_in_i,      # [b, head, seq_in, feature]
    x_out_r, x_out_i,    # [b, head, seq_out, feature]
    pe_in_r, pe_in_i,    # 位置编码
    pe_out_r, pe_out_i
)
```

### 启用MoE
```python
model = TransformerBlock(
    feature_dim=384,
    num_heads=8,
    use_moe=True,
    num_experts=8,
    num_experts_per_token=2  # 每个token使用2个专家
)
```

### 启用KV Cache
```python
model = TransformerBlock(
    feature_dim=384,
    num_heads=8,
    use_kv_cache=True
)

# 第一次推理
out = model(..., use_cache=True)

# 后续推理会复用cache
out = model(..., use_cache=True)

# 清除cache
model.clear_cache()
```

## 设计建议

### 当前实现的优点：
1. ✓ 完整的复数运算（线性层、注意力、归一化）
2. ✓ 高效的MoE实现（只计算选中的专家）
3. ✓ 灵活的KV Cache支持
4. ✓ Gate Score调制机制
5. ✓ 模块化设计，易于扩展

### 可能的改进方向：

#### 1. 复数归一化优化
```python
建议：可以尝试RMSNorm的复数版本
- 更简单的计算
- 可能更稳定的训练
```

#### 2. 位置编码策略
```python
当前：使用加法融合位置编码
建议：可以尝试RoPE的复数版本
- 旋转位置编码
- 可能更好的长度外推性能
```

#### 3. 注意力分数计算
```python
当前：使用复数共轭点积的幅度
替代方案：
- 实部和虚部分别计算注意力后融合
- 使用相位信息增强注意力
```

#### 4. MoE路由优化
```python
建议：
- 添加负载均衡损失
- 实现专家容量限制
- 支持动态专家数量
```

#### 5. 计算效率
```python
优化点：
- 使用torch.complex64原生复数支持
- Flash Attention复数版本
- 梯度检查点降低显存
```

#### 6. 架构变体
```python
可尝试：
- 交错式处理x_in和x_out
- 共享某些权重矩阵
- 添加门控机制决定是否使用cross-attention
```

## 测试验证

代码包含5个测试：
1. 基础功能测试（形状、参数数量）
2. MoE功能验证
3. KV Cache正确性
4. 梯度反向传播
5. 复数性质保持

## 性能特点

- 参数量：约 ~50M (取决于配置)
- 计算量：2倍于实数Transformer（实部+虚部）
- 显存占用：需存储实部和虚部
- MoE可降低激活计算量至 k/n

## 应用场景

此架构特别适合：
- 信号处理（频域特征）
- 相位敏感任务
- 需要旋转不变性的场景
- 多模态融合（x_in和x_out可代表不同模态）
