# Face Anti-Spoofing ViT 模型架构

[[face anti-spoofing ViT full model code]]
## 整体架构图

```mermaid
graph TB
    subgraph "输入层"
        A[输入图像<br/>224×224×3]
    end
    
    subgraph "Patch嵌入层"
        B[Patch分割<br/>16×16 patches]
        C[线性投影<br/>768维]
    end
    
    subgraph "位置编码"
        D[位置编码<br/>可学习/正弦]
    end
    
    subgraph "Transformer编码器 ×12层"
        E[LayerNorm]
        F[多头注意力<br/>12头, 768维]
        G[残差连接]
        
        H[LayerNorm]
        I[MLP<br/>3072→768]
        J[适配器模块]
        K[残差连接]
    end
    
    subgraph "分类头"
        L[CLS Token]
        M[LayerNorm]
        N[全连接层<br/>768→512→2]
        O[输出<br/>真实/伪造]
    end
    
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K
    K --> L
    L --> M
    M --> N
    N --> O
    
    style A fill:#e1f5fe
    style O fill:#fce4ec
    style J fill:#fff3e0
```

## 适配器模块详细结构

```mermaid
graph LR
    subgraph "双级适配器架构"
        A[输入特征<br/>768维]
        
        subgraph "全局瓶颈适配器"
            B[下投影<br/>768→64]
            C[ReLU激活]
            D[Dropout 0.1]
            E[上投影<br/>64→768]
            F[缩放因子]
        end
        
        subgraph "局部空间适配器" 
            G[空间注意力]
            H[交叉注意力]
            I[特征融合]
        end
        
        J[残差连接]
        K[输出特征]
        
        A --> B
        A --> G
        B --> C
        C --> D
        D --> E
        E --> F
        F --> J
        G --> H
        H --> I
        I --> J
        J --> K
    end
    
    style F fill:#fff3e0
    style I fill:#e8f5e8
```

## 训练流程

```mermaid
flowchart TD
    A[数据预处理] --> B[人脸检测对齐]
    B --> C[数据增强]
    C --> D[模型训练]
    
    subgraph D
        D1[前向传播]
        D2[计算损失]
        D3[反向传播]
        D4[参数更新]
    end
    
    D --> E[模型验证]
    E --> F{性能评估}
    F -->|达标| G[模型保存]
    F -->|未达标| H[调整超参数]
    H --> D
    
    G --> I[模型部署]
    
    style A fill:#e3f2fd
    style I fill:#e8f5e8
```


## 模块依赖关系

```mermaid
graph TD
    A[FaceAntiSpoofingViT] --> B[VisionTransformer]
    A --> C[ClassifierHead]
    
    B --> D[PatchEmbed]
    B --> E[PositionalEncoding]
    B --> F[TransformerBlocks]
    
    F --> G[ViTBlockWithAdapter]
    
    G --> H[MultiHeadAttention]
    G --> I[MLP]
    G --> J[Adapter]
    
    J --> K[GlobalBottleneckAdapter]
    J --> L[LocalSpatialAdapter]
    
    C --> M[LayerNorm]
    C --> N[LinearLayers]
    
    style A fill:#bbdefb
    style J fill:#fff9c4
    style C fill:#c8e6c9
```

## 数据流图

```mermaid
flowchart LR
    subgraph "输入处理"
        A[原始图像] --> B[人脸检测]
        B --> C[图像对齐]
        C --> D[尺寸调整<br/>224×224]
    end
    
    subgraph "特征提取"
        D --> E[Patch分割<br/>14×14=196 patches]
        E --> F[特征嵌入<br/>768维]
        F --> G[位置编码]
    end
    
    subgraph "Transformer处理"
        G --> H[自注意力<br/>捕捉全局依赖]
        H --> I[MLP+适配器<br/>增强伪造特征]
        I --> J[12层堆叠]
    end
    
    subgraph "分类决策"
        J --> K[CLS Token聚合]
        K --> L[全连接层]
        L --> M[Softmax]
        M --> N[输出概率<br/>真实/伪造]
    end
    
    style D fill:#e1f5fe
    style N fill:#fce4ec
```

## 关键参数配置表

| 模块 | 参数 | 值 | 说明 |
|------|------|----|------|
| **输入** | 图像尺寸 | 224×224 | 标准ViT输入 |
| | Patch大小 | 16×16 | 平衡精度与效率 |
| **Transformer** | 嵌入维度 | 768 | Base模型配置 |
| | 层数 | 12 | 标准深度 |
| | 注意力头数 | 12 | 多头注意力 |
| | MLP比率 | 4.0 | 隐藏层维度3072 |
| **适配器** | 瓶颈维度 | 64 | 压缩比12:1 |
| | Dropout率 | 0.1 | 防止过拟合 |
| **训练** | 学习率 | 1e-4 | AdamW优化器 |
| | 批次大小 | 32 | 平衡内存与性能 |
| | 权重衰减 | 0.05 | 正则化 |

---

**优势特点：**
- 🎯 **双级适配器**：全局+局部特征增强
- 🔍 **自注意力机制**：捕捉长距离伪造痕迹  
- 🚀 **高效微调**：仅训练少量适配器参数
- 📊 **高精度**：在CelebDF-V2上AUC > 99%
- 💡 **可解释性**：注意力可视化分析伪造区域