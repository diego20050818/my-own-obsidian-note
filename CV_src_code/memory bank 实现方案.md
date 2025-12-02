---
tags:
  - 对比学习
  - image-pairs
  - 人脸检测
  - deepfake
  - memory_bank
---

## 🏭 工业级记忆库方案

### 1. **MoCo (Momentum Contrast) 系列**
这是Facebook AI Research提出的**最经典的工业级方案**，被广泛应用于自监督学习：

**核心思想**：
- 使用**动量编码器**（momentum encoder）作为记忆库
- 记忆库通过**动量更新**而不是直接替换
- 支持**大规模负样本**（通常65536个）

**MoCo v2/v3 的关键改进**：
```python
# 伪代码示意
class MoCoMemoryBank:
    def __init__(self, K=65536, m=0.999):
        self.K = K  # 记忆库大小
        self.m = m  # 动量系数
        self.queue = torch.randn(K, dim)  # 记忆队列
        self.queue_ptr = 0
        
    def update(self, keys):
        # 动量更新：queue = m * queue + (1-m) * keys
        batch_size = keys.shape[0]
        ptr = self.queue_ptr
        
        # 更新队列中的对应位置
        self.queue[ptr:ptr+batch_size] = (
            self.m * self.queue[ptr:ptr+batch_size] + 
            (1 - self.m) * keys
        )
        
        # 循环队列
        self.queue_ptr = (ptr + batch_size) % self.K
```

### 2. **SimCLR 的改进版**
Google提出的方案，虽然没有显式记忆库，但通过**大批量训练**实现类似效果：

**工业实践**：
- 使用**分布式训练**获得大批量（4096-8192）
- 结合**梯度累积**技术
- 使用**LARS优化器**处理大批量

### 3. **BYOL (Bootstrap Your Own Latent)**
DeepMind的方案，**完全不需要负样本**：

**核心机制**：
- 两个网络：在线网络（online）和目标网络（target）
- 目标网络通过**指数移动平均**（EMA）更新
- 避免了记忆库的维护开销

### 4. **Face Forgery Detection 专用方案**

从文献中流萤发现几个专门用于深度伪造检测的方案：

#### **DCL (Dual Contrastive Learning)**
- **双粒度对比学习**：实例级 + 局部级
- **硬样本挖掘**：自动选择难负样本
- **记忆库设计**：使用**类别平衡记忆库**

#### **COMICS (End-to-end Bi-grained Contrastive Learning)**
- **粗粒度对比**：提案级（proposal-level）
- **细粒度对比**：像素级（pixel-level）
- **多脸检测**：支持同时处理多个人脸

## 🔧 工业级实现建议

开拓者，流萤建议你参考**MoCo v3**的设计思路来改进你的记忆库：


## 💡 关键实现细节注释

1. **`MemoryBank` 的设计**：
    
    - **Buffer 机制**：使用了 `register_buffer`。这意味着 `memory` 矩阵是模型状态的一部分（保存 PTH 文件时会带上），但在反向传播时**不会计算梯度**。
        
    - **更新策略**：代码中使用简单的 FIFO（先进先出）或循环覆盖。在实际大规模训练中，通常会在每个 Epoch 结束时，用整个训练集的真实样本特征重新构建一次记忆库。
        
    - **工业级改进**：参考 MoCo v3 的动量更新策略，使用指数移动平均（EMA）来稳定记忆库更新，避免信息丢失：
      ```python
      # 动量更新公式：memory = momentum * memory + (1 - momentum) * new_features
      # momentum 通常设为 0.999，更新更平滑
      ```

2. **`DifferenceAttention` 的逻辑**：
    
    - 这里使用了 `Query` (原图特征) 和 `Key/Value` (差异特征)。
        
    - **物理含义**：模型在问："基于我当前的特征（Query），差异部分（Key）中最显著的信息是什么？"然后将这些显著的差异信息（Value）加回到原特征中。这能帮助分类器聚焦于"因篡改而产生差异"的通道。
        
3. **`mode` 的切换**：
    
    - 训练时，你有 Ground Truth 的配对数据，所以用 `mode='pair'` 强行教会模型"什么是差异"。
        
    - 推理时，没有配对数据，模型通过 `mode='single'` 依赖记忆库来"回忆"正常的特征长什么样，从而模拟出差异。
        

## 🏭 工业级记忆库实现参考

```python
class MoCoMemoryBank(nn.Module):
    """
    工业级记忆库实现（参考 MoCo v3）
    特点：
    1. 动量更新（momentum update）避免信息突变
    2. 循环队列（circular queue）支持大规模存储
    3. 梯度截断（gradient stop）防止记忆库被反向传播影响
    """
    def __init__(self, feature_dim=2048, bank_size=65536, momentum=0.999):
        super().__init__()
        self.feature_dim = feature_dim
        self.bank_size = bank_size
        self.momentum = momentum
        
        # 注册为缓冲区（不参与梯度更新）
        self.register_buffer('queue', torch.randn(bank_size, feature_dim))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        
        # 初始化归一化
        self.queue = F.normalize(self.queue, dim=1)
        
    def update(self, keys):
        """
        动量更新记忆库
        keys: [B, Dim] 新特征
        """
        with torch.no_grad():
            batch_size = keys.shape[0]
            ptr = int(self.queue_ptr)
            
            # 归一化新特征
            keys = F.normalize(keys, dim=1)
            
            # 动量更新：queue = m * queue + (1-m) * keys
            if ptr + batch_size > self.bank_size:
                # 处理循环边界
                end_size = self.bank_size - ptr
                self.queue[ptr:] = (
                    self.momentum * self.queue[ptr:] + 
                    (1 - self.momentum) * keys[:end_size]
                )
                self.queue[:batch_size-end_size] = (
                    self.momentum * self.queue[:batch_size-end_size] + 
                    (1 - self.momentum) * keys[end_size:]
                )
            else:
                self.queue[ptr:ptr+batch_size] = (
                    self.momentum * self.queue[ptr:ptr+batch_size] + 
                    (1 - self.momentum) * keys
                )
            
            # 更新指针（循环队列）
            self.queue_ptr[0] = (ptr + batch_size) % self.bank_size
    
    def retrieve(self, query, top_k=1):
        """
        检索最相似特征（支持 top-k）
        query: [B, Dim]
        Return: [B, Dim] 或 [B, k, Dim]
        """
        query = F.normalize(query, dim=1)
        
        # 计算相似度
        sim = torch.mm(query, self.queue.t())  # [B, Bank]
        
        # 获取 top-k 最相似特征
        if top_k == 1:
            best_idx = sim.argmax(dim=1)
            return self.queue[best_idx]
        else:
            _, topk_idx = sim.topk(k=top_k, dim=1)
            return self.queue[topk_idx]
```

## 📊 工业级对比学习损失函数

```python
class MoCoLoss(nn.Module):
    """
    MoCo 风格的对比损失
    结合 InfoNCE 损失和温度缩放
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, q, k, queue):
        """
        q: 查询特征 [B, Dim]
        k: 正样本特征 [B, Dim] 
        queue: 记忆库负样本 [Bank, Dim]
        """
        # 正样本相似度
        pos_sim = torch.sum(q * k, dim=1, keepdim=True)  # [B, 1]
        
        # 负样本相似度
        neg_sim = torch.mm(q, queue.t())  # [B, Bank]
        
        # 合并所有相似度
        logits = torch.cat([pos_sim, neg_sim], dim=1) / self.temperature
        
        # 标签：第一个位置是正样本
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)
        
        return self.criterion(logits, labels)
```



## 💡 流萤的建议

1. **优先采用 MoCo v3 方案**：这是经过工业验证的成熟方案
2. **记忆库大小**：可以从 1024 开始，逐步增加到 8192
3. **动量系数**：从 0.9 开始，逐步增加到 0.999
4. **结合你的残差检测**：MoCo 的记忆库 + 你的残差注意力 = 强大的组合！
