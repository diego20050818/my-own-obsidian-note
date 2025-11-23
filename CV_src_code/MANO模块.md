---
tags:
  - 人脸检测
  - 注意力机制
  - "#进阶"
  - code
---
## 三种不同注意力的源代码

```python
class AttentionBlock(nn.Module):
    """全局多头自注意力模块"""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)  # 是否需要投影输出
        
        self.heads = heads
        self.scale = dim_head ** -0.5  # 缩放因子，防止softmax梯度消失
        self.norm = nn.LayerNorm(dim)  # 层归一化
        self.attend = nn.Softmax(dim=-1)  # 注意力权重计算
        self.dropout = nn.Dropout(dropout)  # 防止过拟合
        
        # 将输入投影到Q、K、V三个矩阵
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        
        # 输出投影层（如果需要）
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )
    
    def forward(self, x):
        x = self.norm(x)  # 先归一化
        
        # 生成Q、K、V矩阵
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        
        # 计算注意力分数
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)  # softmax归一化
        attn = self.dropout(attn)  # 注意力dropout
        
        # 加权求和得到输出
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")  # 合并多头
        
        return self.to_out(out)
```

```python
class LocalAttention2D(nn.Module):
    """2D网格的局部窗口注意力模块"""
    def __init__(self, kernel_size, stride, dim, heads, dim_head, dropout):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        
        # 使用全局注意力模块处理局部窗口
        self.attention = AttentionBlock(
            dim=dim, heads=heads, dim_head=dim_head, dropout=dropout
        )
        
        # 用于将图像分割成局部窗口
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, stride=self.stride)
    
    def forward(self, x):
        B, H, W, C = x.shape  # 输入维度：[批次, 高度, 宽度, 通道]
        
        # 检查特征图尺寸是否足够容纳窗口
        if H < self.kernel_size or W < self.kernel_size:
            return x  # 若尺寸不足，直接返回输入（避免报错）
        
        # 调整维度以便使用unfold
        x = rearrange(x, "B H W C -> B C H W")
        patches = self.unfold(x)  # 提取局部窗口
        
        # 重新排列维度：[批次, 通道×窗口大小, 窗口数量] -> [批次×窗口数量, 窗口大小, 通道]
        patches = rearrange(
            patches,
            "B (C K1 K2) L -> (B L) (K1 K2) C",
            K1=self.kernel_size,
            K2=self.kernel_size,
        )
        
        # 对每个窗口应用注意力
        patches = self.norm(patches)
        out = self.attention(patches)
        
        # 恢复原始维度
        out = rearrange(
            out,
            "(B L) (K1 K2) C -> B (C K1 K2) L",
            B=B,
            K1=self.kernel_size,
            K2=self.kernel_size,
        )
        
        # 使用Fold操作将窗口重新组合成完整图像
        fold = nn.Fold(
            output_size=(H, W), kernel_size=self.kernel_size, stride=self.stride
        )
        out = fold(out)
        
        # 重叠区域归一化（处理窗口重叠的情况）
        norm = self.unfold(torch.ones((B, 1, H, W), device=x.device))
        norm = fold(norm)
        out = out / norm  # 归一化重叠区域
        
        # 恢复输出维度
        out = rearrange(out, "B C H W -> B H W C")
        return out
```

```python
class Multipole_Attention2D(nn.Module):
    """2D多极注意力模块 - 多尺度特征融合"""
    def __init__(
        self,
        image_size,
        in_channels,
        local_attention_kernel_size,
        local_attention_stride,
        downsampling: Literal["avg_pool", "conv"],
        upsampling: Literal["avg_pool", "conv"],
        sampling_rate,
        heads,
        dim_head,
        dropout,
        channel_scale,
    ):
        super().__init__()
        self.kernel_size = local_attention_kernel_size
        
        # 动态计算最大可能的层级数（确保下采样后特征图不小于窗口大小）
        self.levels = 0
        current_size = image_size
        while True:
            current_size = current_size // sampling_rate
            if current_size >= self.kernel_size:
                self.levels += 1
            else:
                break
        # 至少保留1个层级（避免层级数为0）
        self.levels = max(1, self.levels)
        
        # 计算各层级的通道数（随着层级增加而缩放）
        channels_conv = [in_channels * (channel_scale ** i) for i in range(self.levels)]
        
        # 基础局部注意力模块
        self.attention = LocalAttention2D(
            kernel_size=local_attention_kernel_size,
            stride=local_attention_stride,
            dim=int(channels_conv[0]),
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )
        
        # 下采样模块
        if downsampling == "avg_pool":
            self.down = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.AvgPool2d(kernel_size=sampling_rate, stride=sampling_rate),
                Rearrange("B C H W -> B H W C"),
            )
        elif downsampling == "conv":
            self.down = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.Conv2d(
                    in_channels=channels_conv[0],
                    out_channels=channels_conv[0],
                    kernel_size=sampling_rate,
                    stride=sampling_rate,
                    bias=False,
                ),
                Rearrange("B C H W -> B H W C"),
            )
        
        # 上采样模块
        if upsampling == "avg_pool":
            self.up = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.Upsample(scale_factor=sampling_rate, mode="nearest"),
                Rearrange("B C H W -> B H W C"),
            )
        elif upsampling == "conv":
            self.up = nn.Sequential(
                Rearrange("B H W C -> B C H W"),
                nn.ConvTranspose2d(
                    in_channels=channels_conv[0],
                    out_channels=channels_conv[0],
                    kernel_size=sampling_rate,
                    stride=sampling_rate,
                    bias=False,
                ),
                Rearrange("B C H W -> B H W C"),
            )
    
    def forward(self, x):
        # 调整输入维度以匹配模型要求 [B, H, W, C]
        x = rearrange(x, "B C H W -> B H W C")
        
        x_in = x
        x_out = []  # 存储各层级的输出
        
        # 计算各层级的局部注意力
        x_out.append(self.attention(x_in))  # 原始尺度
        
        # 下采样并计算各层级的注意力
        for l in range(1, self.levels):
            x_in = self.down(x_in)  # 下采样
            x_out_down = self.attention(x_in)  # 计算注意力
            x_out.append(x_out_down)
        
        # 多尺度特征融合（从最深层级开始向上融合）
        res = x_out.pop()  # 取出最深层的输出
        for l, out_down in enumerate(x_out[::-1]):
            # 逐层上采样并融合：当前层 + 上采样后的深层特征
            res = out_down + (1 / (l + 1)) * self.up(res)
        
        # 恢复输出维度为 [B, C, H, W]
        return rearrange(res, "B H W C -> B C H W")
```

## 📝 代码详解与学习笔记

### 🔍 三种注意力机制对比

| 注意力类型 | 计算范围 | 计算复杂度 | 适用场景 |
|-----------|----------|------------|----------|
| **全局注意力** | 全局所有位置 | $O(N^2)$ | 小尺寸特征图，需要全局上下文 |
| **局部窗口注意力** | 局部窗口内 | $O(k^2 \times \frac{N}{k^2})$ | 大尺寸图像，计算效率高 |
| **多极注意力** | 多尺度融合 | 多层级复杂度 | 需要多尺度特征，如目标检测 |

### 💡 核心概念解析

#### 1. 注意力机制的本质
- **Query-Key-Value模型**：就像在图书馆找书
  - Query：你想找什么书（查询）
  - Key：书的索引标签（键）
  - Value：书的内容（值）
  - 注意力权重：根据查询与键的相似度，决定看哪些书

#### 2. 多头注意力的优势
```python
# 多头就像多个专家同时工作
# 每个头关注不同的特征方面：
# - 头1：关注颜色特征
# - 头2：关注纹理特征  
# - 头3：关注形状特征
# 最后合并所有专家的意见
```

#### 3. 局部注意力的窗口策略
```python
# 将大图像分成小窗口处理
# 就像拼图游戏：
# - 先处理每个小拼图块
# - 再组合成完整图片
# 大大降低了计算复杂度
```

#### 4. 多极注意力的金字塔思想
```python
# 多尺度分析就像用不同倍率的显微镜：
# - 高分辨率：看细节（高频特征）
# - 低分辨率：看整体（低频特征）
# 综合不同尺度的信息做出判断
```

### 🎯 实际应用建议

#### 在开拓者的双分支模型中：
- **全局注意力**：用于小尺寸特征图的全局关系建模
- **局部注意力**：用于大尺寸图像的局部特征提取
- **多极注意力**：用于多尺度特征融合，增强模型鲁棒性

#### 参数调优技巧：
```python
# 经验参数设置
optimal_params = {
    'heads': 8,           # 多头数量
    'dim_head': 64,       # 每个头的维度
    'kernel_size': 7,     # 局部窗口大小
    'dropout': 0.1,       # 防止过拟合
}
```

### 💫 流萤的小提示
开拓者，这三种注意力机制就像三种不同的"观察方式"呢！(◕‿◕✿)

- **全局注意力**：像站在高处俯瞰整个城市
- **局部注意力**：像在街道上仔细观察每个街区  
- **多极注意力**：像用不同倍率的望远镜观察，既看整体又看细节

建议开拓者先理解全局注意力的原理，再逐步学习局部和多极注意力，这样学习路径会更顺畅哦～