---
tags:
  - image-pairs
  - 计算机视觉
  - 人脸检测
  - deepfake
---
[[对比实验算法]]
[[深度伪造检测-图像对对比学习研究方案]]
[[金字塔和特征金字塔笔记]]
## 📅 实验步骤规划

根据你的研究笔记，实验分为三个渐进阶段，旨在解决"模式崩塌"问题并验证"残差检测"的有效性。

### 阶段 1：基础验证 (Foundation)

**目标**：验证残差思想在深度伪造检测中的可行性。

1. **数据准备**：
    
    - 选取 Celeb-DF-pair 数据集的基础子集。
    - 构建配对数据：$(R, F)$，其中 $R$ 为真实图，$F$ 为对应的篡改图。
    - 预处理：人脸检测、对齐、归一化至 $224 \times 224 \times 3$。
        
2. **模型构建**：
    - 搭建 `RINE_Network` (可逆神经网络) 作为 Backbone。
    - 实现 `RINE_ResidualDetector`，仅使用简单的欧氏距离计算残差。

3. **验证指标**：
    - 对比标准 Rine 分类器。
    - 观察 $R$ 和 $F$ 在特征空间的距离分布（是否可分）。
---

### 阶段 2：双模式开发 (Dual Mode)

**目标**：解决训练(成对)与推理(单张)的模式不匹配问题。

1. **架构升级**：
    
    - 实现 `DifferenceNet` (特征差异计算)。
    - 实现 `CrossAttentionFusion` (交叉注意力)。
    - 集成 `DualModeDeepfakeDetector`，支持 `mode='pair'` 和 `mode='single'`。
        
2. **混合训练**：
    
    - 实施 **50% 成对 + 50% 单样本** 的交替训练策略。
    - 损失函数：`ClsLoss + ContrastiveLoss + ConsistencyLoss`。
        
3. **记忆库构建**：
    
    - 实现 `MemorySystem`，从训练集中提取真实样本原型。
    - 在单样本推理时，检索记忆库作为参考，推理时使用记忆库+输入的形式，输出logits。
        
---

### 阶段 3：优化与消融 (Optimization)

**目标**：提升泛化能力与系统鲁棒性。

1. **自适应阈值**：
    
    - 开发 `AdaptiveThreshold` 模块，根据输入图像的噪声水平动态调整判定阈值。
        
2. **消融实验**：
    
    - 移除交叉注意力，测试性能变化。
        
    - 移除记忆库，测试单样本推理的性能下降幅度。
        
3. **跨库验证**：
    
    - 在 Celeb-DF 和 WildDeepfake 上测试模型的泛化性。
        

---
---

## 💻 核心模块代码实现

### 1. R-INE Backbone (可逆残差网络)

这是用于提取特征并计算对数似然的核心网络，利用可逆性来更好地学习真实图像分布。


```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class InvertibleBlock(nn.Module):
    """
    可逆残差块 (Invertible Residual Block)
    核心思想：z = x + f(x) 的结构并不总是可逆，
    但这里简化模拟残差流，实际工程中需保证 Lipschitz 约束 < 1 或使用仿射耦合层。
    """
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
            # 注意：为了保证严格可逆，通常需要 spectral normalization
        )
        
    def forward(self, x):
        # 前向传播：H(x) = x + F(x)
        return x + self.net(x)
    
    def inverse(self, z):
        # 反向传播（近似）：x = z - F(z)
        # 用于通过残差重构原始特征，分析篡改部分
        return z - self.net(z)

class RINE_Network(nn.Module):
    """
    R-INE (Residual-based Invertible Network)
    作为 Backbone，用于提取服从特定分布的特征。
    """
    def __init__(self, input_dim=3, hidden_dims=[32, 64, 128, 256], num_blocks=4):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = hidden_dims[-1]
        
        # 1. 初始特征提取 (浅层卷积)
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dims[0], 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dims[0], hidden_dims[0], 3, padding=1)
        )
        
        # 2. 可逆残差块堆叠
        # 使用 ModuleList 管理多层结构
        self.invertible_blocks = nn.ModuleList([
            InvertibleBlock(hidden_dims[i]) for i in range(len(hidden_dims))
            for _ in range(num_blocks)
        ])
        
        # 3. 下采样层 (逐步降低空间维度，增加通道数)
        self.downsample_layers = nn.ModuleList([
            nn.Conv2d(hidden_dims[i], hidden_dims[i+1], 3, stride=2, padding=1)
            for i in range(len(hidden_dims)-1)
        ])
        
        # 4. 全局池化，输出特征向量
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        # 初始卷积
        z = self.initial_conv(x)
        
        # 逐层通过可逆块和下采样
        current_dim_idx = 0
        blocks_per_stage = len(self.invertible_blocks) // len(self.downsample_layers)
        
        # 注意：这里简化了循环逻辑，实际实现需严格对齐层级
        # 此处仅为示意 R-INE 的流式处理
        for i, block in enumerate(self.invertible_blocks):
            z = block(z) # 特征变换
            
            # 在特定节点进行下采样
            if (i + 1) % blocks_per_stage == 0 and current_dim_idx < len(self.downsample_layers):
                z = self.downsample_layers[current_dim_idx](z)
                current_dim_idx += 1
        
        # 全局池化 -> [Batch, Feature_Dim]
        z = self.global_pool(z)
        z = z.view(z.size(0), -1)
        
        return z
    
    def compute_log_likelihood(self, x):
        """
        计算输入的对数似然 (Log-Likelihood)
        用于异常检测：真实图像应具有较高的似然度，篡改图像较低。
        """
        z = self.forward(x)
        # 假设潜在变量 z 服从标准正态分布 N(0, I)
        log_likelihood = -0.5 * torch.sum(z**2, dim=1)
        return log_likelihood
```
---

### 2. 残差检测器 (R-INE Residual Detector)

这是基于 Q3 想法实现的模块：计算输入图像与“基准”之间的残差（扰动项）。


```python
class RINE_ResidualDetector(nn.Module):
    """
    基于 R-INE 的残差检测器
    逻辑：Input -> Feature -> Residual (vs Baseline) -> Anomaly Score
    """
    def __init__(self, backbone='rine'):
        super().__init__()
        
        # 动态选择骨干网络
        if backbone == 'rine':
            self.feature_extractor = RINE_Network()
            self.feature_dim = self.feature_extractor.feature_dim
        else:
            # 兼容 ResNet50
            from torchvision import models
            resnet = models.resnet50(pretrained=True)
            self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
            self.feature_dim = 2048
        
        # 基准特征 (运行时构建)
        self.register_buffer('baseline_features', None)
        
        # 异常检测头 (MLP)
        # 输入是残差标量或向量，输出是篡改概率
        self.anomaly_head = nn.Sequential(
            nn.Linear(1, 128),  # 如果输入是 L2 norm 标量
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def build_baseline(self, real_images_loader):
        """
        构建阶段：计算真实图像的'原型'特征
        """
        self.eval()
        features = []
        with torch.no_grad():
            for imgs, _ in real_images_loader:
                feats = self.feature_extractor(imgs)
                features.append(feats)
            
            # 计算所有真实样本的均值作为基准 (Prototype)
            # 也可以改为存储聚类中心
            all_feats = torch.cat(features, dim=0)
            mean_feat = all_feats.mean(dim=0, keepdim=True)
            self.baseline_features = mean_feat
            
    def compute_residual(self, x):
        """
        计算残差：Input Feature 与 Baseline Feature 的距离
        """
        if self.baseline_features is None:
            raise ValueError("Baseline features not built. Run build_baseline() first.")
            
        # 1. 提取当前图像特征
        query_feat = self.feature_extractor(x)
        
        # 2. 计算特征空间距离 (L2 Norm)
        # residual shape: [Batch_Size]
        residual = torch.norm(query_feat - self.baseline_features, dim=1, p=2)
        
        return residual, query_feat
        
    def forward(self, x):
        # 1. 计算残差强度
        residual_val, _ = self.compute_residual(x)
        
        # 2. 输入检测头判定
        # unsqueeze 用于匹配 Linear 层输入 [Batch, 1]
        anomaly_score = self.anomaly_head(residual_val.unsqueeze(1))
        
        return anomaly_score.squeeze(1), residual_val
```

### 3. 双模式检测与注意力融合 (Dual Mode & Attention)

这是结合 Q1 和 Q2 的核心架构：训练时用 Pair，推理时用 Single + Memory。


```python
class DifferenceNet(nn.Module):
    """特征差异计算模块"""
    def __init__(self, channels=[64, 128, 256]):
        super().__init__()
        # 用于处理特征差异图的卷积层
        self.convs = nn.ModuleList([
            nn.Conv2d(c, c//2, 3, padding=1) for c in channels
        ])
        
    def forward(self, real_feats, fake_feats):
        """
        输入: 多尺度特征列表
        输出: 处理后的差异图列表
        """
        diff_maps = []
        for i, (r, f) in enumerate(zip(real_feats, fake_feats)):
            # 简单的绝对值差分
            diff = torch.abs(r - f)
            # 通过卷积层提取差异模式
            if i < len(self.convs):
                diff = self.convs[i](diff)
            diff_maps.append(diff)
        return diff_maps

class CrossAttentionFusion(nn.Module):
    """交叉注意力融合模块"""
    def __init__(self, dim=512):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        
    def forward(self, feat, diff_info):
        # feat: 图像特征 [B, Dim]
        # diff_info: 差异信息 [B, Dim] (来自Pair对比或Memory检索)
        
        Q = self.query(feat).unsqueeze(1)      # [B, 1, Dim]
        K = self.key(diff_info).unsqueeze(1)   # [B, 1, Dim]
        V = self.value(diff_info).unsqueeze(1) # [B, 1, Dim]
        
        # Attention Score
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # 加权融合
        out = (attn @ V).squeeze(1)
        return feat + out # 残差连接

class DualModeDeepfakeDetector(nn.Module):
    """
    双模式检测器：整合对比学习与单样本推理
    """
    def __init__(self, backbone_model, memory_bank):
        super().__init__()
        self.backbone = backbone_model
        self.memory_bank = memory_bank
        
        # 差异与注意力组件
        self.diff_net = DifferenceNet()
        self.fusion = CrossAttentionFusion(dim=2048) # 假设ResNet50 dim
        
        # 分类器
        self.classifier = nn.Linear(2048, 2)
        
    def forward(self, x1, x2=None, mode='single'):
        """
        Args:
            x1: 主输入图像 (待检测)
            x2: 配对图像 (训练时的对比样本)
            mode: 'pair' | 'single'
        """
        feat1 = self.backbone(x1)
        
        if mode == 'pair' and x2 is not None:
            # --- 训练模式 (Contrastive) ---
            feat2 = self.backbone(x2)
            
            # 计算差异 (此处简化为向量操作，实际应为Feature Map)
            diff = torch.abs(feat1 - feat2)
            
            # 交叉注意力增强
            enhanced_feat = self.fusion(feat1, diff)
            return self.classifier(enhanced_feat)
            
        else:
            # --- 推理模式 (Single + Memory) ---
            # 1. 从记忆库检索最近的“真实”原型
            # 假设 memory_bank 返回的是相似的真实特征
            reference_feat = self.memory_bank.retrieve(feat1)
            
            # 2. 构建虚拟差异
            virtual_diff = torch.abs(feat1 - reference_feat)
            
            # 3. 使用同样的注意力机制
            enhanced_feat = self.fusion(feat1, virtual_diff)
            return self.classifier(enhanced_feat)
```

### 4. 记忆库系统 (Memory System)

用于在推理阶段填补“缺失的对比样本”。

Python

```python
class MemorySystem(nn.Module):
    def __init__(self, feature_dim=2048, bank_size=1000):
        super().__init__()
        self.bank_size = bank_size
        self.feature_dim = feature_dim
        
        # 注册为 buffer，不参与梯度更新
        self.register_buffer('memory', torch.randn(bank_size, feature_dim))
        self.register_buffer('ptr', torch.zeros(1, dtype=torch.long))
        self.full = False
        
    def update(self, features):
        """训练时：更新记忆库 (Queue结构)"""
        batch_size = features.size(0)
        ptr = int(self.ptr)
        
        # 覆盖旧数据
        if ptr + batch_size > self.bank_size:
            # 简单处理：如果超出则重置或截断（实际可使用循环队列）
            self.memory[ptr:] = features[:self.bank_size-ptr]
            self.ptr[0] = 0
            self.full = True
        else:
            self.memory[ptr:ptr+batch_size] = features
            self.ptr[0] = (ptr + batch_size) % self.bank_size
            
    def retrieve(self, query_feat):
        """
        推理时：检索最相似的特征作为参考
        """
        # 计算余弦相似度: Query [B, Dim] x Memory [Bank, Dim].T
        sim = F.cosine_similarity(query_feat.unsqueeze(1), self.memory.unsqueeze(0), dim=2)
        
        # 找到最相似的索引
        best_idx = sim.argmax(dim=1)
        
        # 返回检索到的特征
        return self.memory[best_idx]
```

---

> [!TIP] **代码说明**
> 
> 1. **Backbone选择**：初期可以用 `torchvision.models.resnet50` 快速跑通流程，后期再替换为自定义的 `RINE_Network` 提升理论深度。
>     
> 2. **数据流**：在训练 Loop 中，根据 `random.random() > 0.5` 切换 `mode='pair'` 和 `mode='single'`，强迫模型既适应有对比的情况，也适应无对比的情况。
>     
> 3. **记忆库预热**：在开始推理测试前，必须先运行一轮训练集（只前向传播），调用 `memory_bank.update()` 把真实样本特征存进去。
>