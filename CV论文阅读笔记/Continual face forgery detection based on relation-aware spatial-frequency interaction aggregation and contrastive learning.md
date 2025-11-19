关系感知空频交互聚合与对比学习的持续性人脸伪造检测
## 空频
空——空间域，直观看到的图像
频——频域，描述图像不同频率成分的强度

## 关系感知
不仅仅关注单个响度或者单个区域的特征，而是不同部分之间的关系
1. 空间关系：分析人脸不同区域之间的相对位置、比例等看看有没有不协调的地方
2. 分析不同区域的频率特征是否保持一致，会不会出现一些地方锐利一些地方平滑的情况
3. 跨域关系：同时分析空间阈特征和频率之间的关联，
4. 真实的区域应该其空间上的纹理与其频率由内在关联，伪造的区域有可能破坏了这种内在的一致性

> [!PDF|important] [[Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning.pdf#page=1&selection=293,0,364,10&color=important|Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning, p.1]]
> > To address this challenge, we propose a novel continual face forgery detection (CFFD) framework that integrates multi-view knowledge distillation and a hybrid sampling replay mechanism to improve the generalization of the model for evolving forgery techniques
> 
> 创新点这一块，提出了一个CFFD的框架
> 关键词：多视角知识蒸馏和混合采样的回放机制
> 目的：提升模型对不断演进的伪造技术的泛化能力

## CFFD
![[Pasted image 20251022161909.png]]
核心解决痛点：持续学习
### CFFD框架的核心组件

1. **持续学习场景设定 (Continual Learning Scenario)**
    
    - 模型不是一次性接收所有数据，而是按**时间序列**或**任务序列**接收数据。
    - 例如，第1阶段：学习检测Method A和Method B生成的伪造人脸。
    - 第2阶段：出现新方法Method C，模型需要在学习Method C的同时，保持对A和B的检测能力。
    - 关键挑战是**灾难性遗忘（Catastrophic Forgetting）**：模型在学习新任务时，会覆盖或遗忘旧任务的知识。
2. **混合采样回放机制 (Hybrid Sampling Replay Mechanism)**
    
    - **目的**：缓解灾难性遗忘。
    - **原理**：为了不让模型忘记旧知识，需要在学习新任务时，“复习”一些旧任务的样本。
    - **“混合采样”** 指的是回放样本的选取策略不是单一的，而是结合了多种方法：
        - **基于代表性的采样**：从每个旧任务中选择最能==代表该任务特征==的样本（例如，==特征空间中的聚类中心==）。
        - **基于不确定性的采样**：选择模型对旧任务==预测最不确定的样本==，这些样本对巩固知识最有帮助。
        - **基于多样性的采样**：确保回放的样本覆盖旧任务的不同子类型或变化。
        - **可能还包括真实样本**：为了保持对真实人脸的判别能力，回放机制也可能包含少量真实人脸样本。
    - **优势**：相比于简单的随机回放或固定大小的回放缓冲区，混合采样能更高效地利用有限的存储空间，选择出最有价值的“复习资料”，从而更有效地防止遗忘。
3. **多视角知识蒸馏 (Multi-View Knowledge Distillation)**
    
    - **目的**：在不直接访问旧数据（或仅访问回放样本）的情况下，保护旧知识。
    - **原理**：
        - 在学习新任务之前，先将当前的模型（称为**学生模型**）冻结，作为**教师模型**。
        - 当学生模型学习新任务的新数据时，它不仅要拟合新数据的标签，还要**模仿教师模型对旧任务数据（或回放数据）的输出**。
        - 这种“模仿”通常通过最小化教师模型和学生模型在这些数据上的输出（如分类概率、特征表示）之间的差异（如KL散度、均方误差）来实现。
    - **“多视角”** 的含义：
        - 蒸馏不仅仅发生在最终的分类输出层。
        - 它可能同时发生在**多个层次**（如中间特征层、注意力图），或者**多个分支**（如空间分支、频率分支、关系感知模块的输出）。
        - 例如，在您之前提到的“关系感知空频交互”模型中，“多视角”可能指同时对空间特征、频率特征以及它们交互后的特征进行知识蒸馏。
    - **优势**：多视角蒸馏能更全面地保留教师模型学到的复杂知识（包括特征表示、内部关系等），而不仅仅是最终的分类决策，从而提供更强的抗遗忘能力。
> [!PDF|important] [[Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning.pdf#page=2&selection=555,0,581,10&color=important|Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning, p.2]]
> >  In this framework, we further propose a relation-aware spatial-frequency interaction aggregation network (RSIA-Net)
> 
> 该网络利用所设计的关系感知空间-频率交互聚合（RSIA）模块，在空间域和频率域特征之间基于相关性指导进行分层交互，实现有效的互补嵌入，有助于提取更细粒度的伪造线索。此外，我们设计了一种新颖的分层空间-频率对比学习（HSCL）机制，以进一步增强空间-频率信息的融合与对齐效果，并提升特征的判别能力。
> RSIA是CFFD持续学习框架的内部使用的主干网络、核心特征提取器
>

>[!PDF|有用的信息] [[Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning.pdf#page=2&selection=1063,0,1143,2&color=有用的信息|Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning, p.2]]
> > Existing methods for detecting face forgery can be generally categorized into heuristic methods and deep learning-based methods. Heuristic approaches aim to detect forgeries by extracting tampering artifacts utilizing hand-crafted features such as visual [1] or human biometrics [2,3
> 
> 现有的方法分为启发式方法和深度学习的方法
> 启发式：通过手工设计特征提取篡改痕迹来检测伪造内容
> [!PDF|有用的信息] [[Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning.pdf#page=3&selection=704,0,768,9&color=有用的信息|Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning, p.3]]
> > Existing continual learning approaches can be broadly categorized into three groups: replay-based methods [26], regularizationbased methods [27,28], and structure-based methods [29]. In the field of face forgery detection
> 
> 现有持续学习的方法分为三种：
> 基于回放的方法
> 基于正则化的方法
> 基于结构的方法

![[Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning.pdf#page=3&rect=296,456,580,758&color=important|Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning, p.3]]
> [!PDF|important] [[Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning.pdf#page=3&selection=1178,0,1725,2&color=important|Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning, p.3]]
> > The continual face forgery detection task aims to refine the detection model using a limited number of new samples to detect new types of forgery and retain knowledge from previous tasks to prevent catastrophic forgetting. Given a set of face forgery detection task sequences[𝑇 1 𝐷1 , 𝑇 2 𝐷2 , ⋯ , 𝑇 𝐼 𝐷𝑖 ] and 𝑇 1 𝐷1 ∩ 𝑇 2 𝐷2 ⋯ 𝑇 𝐼 𝐷𝑖 = ∅,where I denotes the total number of tasks, 𝑇 𝑖 𝐷𝑖 = {(𝑥𝑖 1, 𝑦1), (𝑥𝑖 2, 𝑦2), ⋯ , (𝑥𝑖 𝑗 , 𝑦𝑗 )}𝐷𝑖 𝑗=1 denotes the i-th task, which contains a total of 𝐷𝑖 real and fake samples, 𝑥𝑖 𝑗 is the j-th image of 𝑇 𝑖 𝐷𝑖 , 𝑦𝑗 ∈ {0, 1} is the corresponding label. In the CFFD framework, the detection model 𝑀 is sequentially trained using the datasets in 𝑇 𝑖 𝐷𝑖 . Let 𝑀𝑖−1 denotes the model obtained by training on task 𝑇 𝑖−1 𝐷𝑖−1 . When learning a new task 𝑇 𝑖 𝐷𝑖 , the objective is to learn new knowledge to update the detection model 𝑀𝑖−1 → 𝑀𝑖 to address the new task 𝑇 𝑖 𝐷𝑖 (𝑖 ≥ 2) based on the previous model 𝑀𝑖−1, the current task 𝑇 𝑖 𝐷𝑖 , and the previous task 𝑇 𝑖−1 𝐷𝑖−1 . The constraints of the model update should be able to maintain the detection performance for both the new task 𝑇 𝑖 𝐷𝑖 and the previous tasks of 𝑇 1 𝐷1 , 𝑇 2 𝐷2 , ⋯ , 𝑇 𝑖−1 𝐷
> 


检测模型在这篇论文中是可以持续不断的更新的，随着更新次数的变化，模型用
$$
M_{1},M_{2},……,M{i-1},M{i}
$$
表示，而每一次更新参数模型，都会用上一次模型作为教师模型进行训练，并且在训练的过程中会把教师模型的参数进行冻结

关于模型的输入，从教师模型和学生模型两个温度进行分析
教师模型：输入经过选择的往期样本，然后经过空域变化、频域变化，并且将空域和频域的特征融合之后进行分类任务

学生模型：除了经过选择的往期样本（要与教师模型的样本保持一致）以外，还要将新伪造方法（新的样本范式）做同样的空域、频域、融合处理之后进行分类任务

> [!PDF|important] [[Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning.pdf#page=3&selection=1760,0,1796,7&color=important|Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning, p.3]]
> > It has three main components: a sample selection mechanism, a face forgery detection model, and a knowledge preservation module.
> 
> 样本选择机制、人脸伪造检测模型和知识保留模块

> [!PDF|important] [[Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning.pdf#page=3&selection=1981,0,2053,2&color=important|Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning, p.3]]
> > The knowledge preservation module uses a multi-perspective distillation mechanism to minimize the difference between the outputs of the teacher’s model 𝑀𝑖−1 (trained on 𝑇 𝑖−1 𝐷′ 𝑖−1 ) and the student’s model 𝑀
> 
> 使用多视角蒸馏机制最小化教师模型和学生模型输出之间的差异
> 减少灾难性遗忘

多视角蒸馏机制相比传统蒸馏机制，除了在模型的最终输出的概率分布（logits或者softmax输出）尽可能接近教师模型的输出之外，还在中间的特征图经过均方误差或者余弦相似度等损失函数最小化他们之间的差异，还有这两个：
关系型蒸馏
样本之间的内在关系或者结构，蒸馏样本对之间的相似性关系，例如在特征空间中，两个真实人脸之间的距离或者向量夹角应该小于一个真实和一个伪造人脸之间的关系
域级蒸馏
不同模态之间的特征相应，比如空间域和频率域
分别对空间域和频率域的特征或者输出进行蒸馏

这张图是论文《A Relation-aware Spatial-Frequency Interaction Aggregation Network for Continual Face Forgery Detection》中提出的 **RSIA-Net** 模型的架构图，它清晰地展示了模型如何融合空间域和频率域信息来检测伪造人脸。我们来逐部分解释。

---
![[Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning.pdf#page=4&rect=26,271,567,771&color=important|Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning, p.4]]

### **整体结构 (a) model framework**

这是整个模型的主框架，分为三个主要部分：

1.  **输入分支**：
    *   **Spatial domain branch (空间域分支)**：直接接收原始伪造图像（Forgery Images）。
    *   **Frequency domain branch (频率域分支)**：首先对输入图像进行 **DCT (离散余弦变换)** 处理，将图像从像素空间转换到频率空间，然后处理频域特征。

2.  **双分支特征提取与交互**：
    *   两个分支都包含一个由多个模块组成的网络，这些模块可能是卷积层、注意力机制等，用于提取多层级的特征。
    *   空间域分支输出一系列特征 $F_s^1$, $F_s^2,$ $F_s^3$, $F_s^4$。
    *   频率域分支输出一系列特征 $F_f^1$, $F_f^2$, $F_f^3$, $F_f^4$。

3.  **核心交互与融合模块**：
    *   **SFIA (Spatial-Frequency Interaction Aggregation)**：位于每个层级的初始交互点。它负责在每一层将空间域和频率域的特征进行初步聚合。
    *   **RSIA (Relation-aware Spatial-frequency Interaction Aggregation)**：这是本文的核心创新模块。它接收来自 SFIA 的输出，并进一步进行更深层次、基于“关系感知”的交互。RSIA 模块会生成一个增强后的特征 $f^l$ (例如 $f^1,$ $f^2$)。
    *   **FAF (Feature Adaptive Fusion)**：在所有层级的 RSIA 输出之后，这个模块负责最终的自适应融合，将所有层次的信息整合成一个单一的、强大的特征表示。

4.  **分类器**：
    *   最终融合后的特征被送入一个分类器（Classifier），输出判断结果：**Real / Fake**。

5.  **损失函数**：
    *   **L_CE**：交叉熵损失，用于监督最终分类任务。
    *   **L_s_SFCL, L_f_SFCL, L_s→f_SFCL, L_f→s_SFCL**：这些是 **HSCL (Hierarchical Spatial-Frequency Contrastive Learning)** 机制引入的对比学习损失。它们作用于不同层级的空间和频率特征上，目的是增强同类型样本（如真实/真实或伪造/伪造）在特征空间中的相似性，同时拉远不同类型样本的距离，从而提升特征的判别能力。

---

### **核心模块详解**

#### **(b) Spatial-Frequency Interaction Aggregation (SFIA)**

这是一个基础的交互聚合模块。

*   **输入**：来自空间域的特征 `F_s^A` 和来自频率域的特征 `F_f^A`。
*   **过程**：
    1.  **计算相关性矩阵**：分别计算空间特征 `F_s^A` 与频率特征 `F_f^A` 之间的相关性（`K_s` 和 `K_f`）。这通常通过计算它们的点积或使用注意力机制实现。
    2.  **生成注意力权重**：将 `K_s` 和 `K_f` 通过 Softmax 函数归一化，得到注意力权重 `w_s` 和 `w_f`。
    3.  **加权融合**：使用 `w_s` 对 `F_f^A` 进行加权（元素级乘法 `⊗`），得到 `V_s`；使用 `w_f` 对 `F_s^A` 进行加权，得到 `V_f`。
    4.  **最终输出**：将 `V_s` 和 `V_f` 相加（元素级求和 `⊕`），得到融合后的特征 `A_s` 和 `A_f`。
*   **目的**：让空间和频率特征相互“关注”对方，进行初步的信息交换。

#### **(c) Relation-aware Spatial-frequency Interaction Aggregation (RSIA)**

这是本文的核心创新模块，比 SFIA 更高级。
## SFIA
SFIA 的主要目的是在模型的**每一层**，实现**空间域特征**（来自原始图像）和**频率域特征**（来自DCT等变换）之间的**初步交互与信息融合**。它通过一种“注意力”或“相关性引导”的机制，让两个不同模态的特征能够相互参考、相互增强。

1. **输入**：
    
    - 空间域特征 `F_s^A`
    - 频率域特征 `F_f^A`
2. **计算跨模态相关性**：
    
    - 计算空间特征 `F_s^A` 与频率特征 `F_f^A` 之间的相关性，得到一个相关性矩阵 `K_s`。
    - 计算频率特征 `F_f^A` 与空间特征 `F_s^A` 之间的相关性，得到另一个相关性矩阵 `K_f`。
    - 这一步相当于让两个模态“互相看一眼”，了解对方哪些部分是相关的。
3. **生成注意力权重**：
    
    - 将 `K_s` 和 `K_f` 分别通过 `Softmax` 函数进行归一化，得到注意力权重 `w_s` 和 `w_f`。
    - 这些权重代表了在融合时，应该“关注”对方特征的哪些部分。
4. **加权融合**：
    
    - 使用 `w_s` 对频率特征 `F_f^A` 进行加权（`⊗` 表示元素级乘法），得到一个“被空间注意力调制过的”频率特征 `V_s`。
    - 使用 `w_f` 对空间特征 `F_s^A` 进行加权，得到一个“被频率注意力调制过的”空间特征 `V_f`。
5. **输出融合结果**：
    
    - 将 `V_s` 和 `V_f` 相加（`⊕` 表示元素级求和），得到最终的融合特征 `A_s` 和 `A_f`。
## RFIA后面的部分

*   **输入**：来自 SFIA 的输出特征 `F_s` 和 `F_f`，以及一个可能来自前一层的特征 `F_p`。
*   **过程**：
    1.  **多尺度特征增强 (MSFE)**：对 `F_s` 和 `F_f` 进行多尺度处理。这里使用了 **D-Conv (可变形卷积)**，其感受野大小 (`r=3, 5, 7`) 不同，可以捕捉不同尺度的上下文信息。处理后的特征经过 `1x1` 卷积调整通道数后，再进行 `3x3` 卷积，最后合并为增强后的特征。
    2.  **特征细化模块 (FRM)**：对 `F_p` 进行处理，包括卷积、批归一化 (BN) 和 ReLU 激活，使其更精细。
    3.  **SFTIA 模块**：这实际上就是 (b) 中的 SFIA 模块，但在这里它处理的是经过 MSFE 和 FRM 增强后的特征。它再次执行空间-频率交互，产生新的聚合特征。
    4.  **最终融合**：将 SFTIA 的输出与其他路径（如 MSFE 的输出）进行融合，最终输出给下一层。
*   **目的**：RSIA 模块通过结合 **多尺度特征增强 (MSFE)** 和 **特征细化 (FRM)**，并利用 **SFIA** 进行交互，实现了更复杂、更有效的跨模态信息融合。它不仅考虑了特征间的相关性，还考虑了特征的局部和全局上下文，从而能提取更细粒度、更具判别性的伪造线索。

---

### **总结**

这张图展示了一个先进的伪造检测网络：

1.  **双分支设计**：同时分析空间和频率域信息，充分利用了两种模态的互补性。
2.  **分层交互**：在多个层级上进行空间-频率交互，逐步提炼信息。
3.  **核心创新 (RSIA)**：通过多尺度处理、特征细化和关系感知交互，实现了深度、高效的特征融合。
4.  **对比学习 (HSCL)**：在训练过程中，利用对比损失来优化特征表示，使模型学得更好。
5.  **自适应融合 (FAF)**：最终将所有层次的精华信息汇聚，为分类器提供最强大的决策依据。

整个框架旨在解决传统方法在融合空间和频率特征时效率低下、信息冗余的问题，从而提升伪造检测的精度和泛化能力。

## 空间-频率域内对比学习SFICL


![[Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning.pdf#page=6&rect=30,448,296,541|Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning, p.6]]
> [!PDF|important] [[Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning.pdf#page=6&selection=255,1,443,6&color=important|Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning, p.6]]
> > hile images with different labels belong to negative samples, i.e., Neg𝑖 = {𝑥𝑘 ∣ 𝑦𝑘 ≠ 𝑦𝑖 }. During model training, the intra-contrastive loss enhances the similarity between positive sample features 𝐹 𝑙−𝑥𝑖 𝑠 and 𝐹 𝑙−𝑥𝑗 𝑠 while reducing the similarity between negative sample features 𝐹 𝑙−𝑥𝑖 𝑠 and 𝐹 𝑙−𝑥𝑘 𝑠 . This pulls positive pairs closer and pushes negative pairs apart to optimize the feature space, enforcing the model to learn intra-class compact and inter-class separable feature distributions within the same domain
> 
> 在单个域之内进行对比，可以让模型发现更加细微的差异，提升特征的判别能力
 损失函数使用了类似熵的思想，通过计算两个域之间特征空间的散点，尽量使不同类别之间的特征散点部落分离的更开（前面的$φ$就是在做这个事情，同类取不同类不取，相互之间比较不能是自己跟自己比（i != j)
 $N(·)$表示的是两个样本之间的余弦相似度
 > T的作用就是temperture，T越大分类结果越平均，熵越高
 > 在这个论文中取的T值为0.01

然后将各自的频域和空域通过这个损失加起来就是完整的Loss
![[Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning.pdf#page=6&rect=30,320,159,340&color=important|Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning, p.6]]

## RSIA对比学习
> [!PDF|有用的信息] [[Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning.pdf#page=6&selection=939,1,985,1&color=有用的信息|Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning, p.6]]
> > ince the contrastive distribution of the SFICL is derived from a single feature domain, the correlations between different domains cannot be captured effectively.
> 
> 局限性：SFIA每一次只能处理一个特征域，没有办法跨域进行联合对比学习，所以提出了跨域对比学习的RSIA

![[Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning.pdf#page=6&rect=302,631,564,723&color=有用的信息|Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning, p.6]]
> [!PDF|important] [[Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning.pdf#page=6&selection=1078,0,1179,1&color=important|Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning, p.6]]
> > For a mini-batch containing 𝑁 samples {𝑥𝑖, 𝑦𝑖 }𝑁−1 𝑖=0 , [𝐹 𝑙_𝑥𝑖 𝑠 ]𝑁−1 𝑖=0 and [𝐹 𝑙_𝑥𝑖 𝑓 ]𝑁−1 𝑖=0 denote the spatial domain and frequency domain features extracted from the 𝑁 samples {𝑥𝑖, 𝑦𝑖 }𝑁−1 𝑖=
> 
> 从每个批次的样本中提取空间域和频率域的特征，用小噗呲中某个空间域的人脸图像$x_{i}$作为锚点，然后在将频率域中类别相同的图像作为正样本，不同的就是负样本

> [!PDF|important] [[Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning.pdf#page=6&selection=1656,0,1708,6&color=important|Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning, p.6]]
> > Note that different domain information for the same sample also belongs to positive sample pairs, so the condition Φ[𝑖≠𝑗] is not set in SFCL.
> 
> 相同样本不同域也是正样本对，所以并没有设置i!=j

![[Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning.pdf#page=6&rect=300,546,454,575&color=有用的信息|Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning, p.6]]
最终使用的是多个层次上空域和频域内对比损失和空间、频域跨域对比的损失进行整合，最终的分层空间频率对比损失在上面

## 持续学习的脸部篡改检测框架CFFD
> [!PDF|important] [[Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning.pdf#page=6&selection=1871,0,2030,11&color=important|Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning, p.6]]
> > Thus, we propose a CFFD framework that can help the model quickly adapt to new forgery techniques using limited new data without training the entire model from scratch. In addition, it balances the learning process by incorporating knowledge preservation and replay mechanisms, enabling the model to promote the learning of new tasks while mitigating the catastrophic forgetting of knowledge from previous tasks. Fig. 4 shows that our CFFD framework employs two mechanisms: multi-perspective knowledge distillation and hybrid sampling replay mechanism.
> 
> 引入只是保留和回放机制平衡学习过程，环节对先前任务知识的灾难性遗忘
> 1. 多视角知识蒸馏
> 2. 混合采样回放机制

### 多视角知识蒸馏

> [!PDF|important] [[Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning.pdf#page=6&selection=2040,0,2094,5&color=important|Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning, p.6]]
> > Multi-perspective knowledge distillation uses soft label knowledge distillation (SLKD) and multi-level feature knowledge distillation (MFKD) to transfer the knowledge from the teacher model to the student model
> 
> 
### 软标签知识蒸馏SLKD
利用教师模型输出的软标签指导学生模型的学习
软标签和硬标签不同的事，硬标签对于这种任务输出0或者1表示真假，但是软标签会输出概率，比如一张图形有90%的概率是真的，10%的概率是假的
通过让学生的输出概率分布尽可能接近教师的输出分布，学生不仅能学会分类还能集成教师对样本之间相似性、不确定性的判断，有助于保留对往期任务的记忆，避免完全忘记之前学过的伪造模型
损失函数直接使用的KL散度进行学生模型和教师模型之间分布的对比
![[Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning.pdf#page=6&rect=306,159,480,288&color=important|Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning, p.6]]
### MFKD多层级特征知识蒸馏
核心思想：不仅模仿教师模型的最终输出，还模仿中间层的特征表示
因为网络的中间层捕捉了不同抽象级别的特征，比如边缘到纹理到部件到全局结构
让学生的中间层特征接近教师的对应层的特征，可以更加深入的传递如何看图的知识，而不仅仅是怎么分类
- 在多个网络层级（如 conv3、conv4、conv5）上，计算学生和教师特征图之间的差异（如 L2 损失或余弦相似度），并加入损失函数。
- 帮助学生模型学习到与教师相似的特征提取能力
- 保留对往期任务的敏感性
![[Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning.pdf#page=7&rect=35,347,177,376&color=important|Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning, p.7]]让学生模型在每一层提取的特征，尽可能接近教师的模型在同一个维度的特征
中间的$f^{x_{{i}}}_{t_{l}}-f^{x_{{i}}}_{s_{l}}$表示教师和学生对同一个样本xi在第l层的特征差异，使用的是逐元素相减
$∥⋅∥^2_{2}$  表示对这个差异向量计算L2范数的平方，得到一个标量，代表差距的大小
$\sum_{{i=1}}^N$  表示对一个batch内的所有样本求和，得到该层的总差异
$\frac{1}{L}\sum_{l=1}^{L}$  表示对所有L个层取平均，得到整体的特征差异

## 样本回放机制
> [!PDF|important] [[Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning.pdf#page=7&selection=322,1,367,1&color=important|Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning, p.7]]
> > he sample replay mechanism simulates a continual learning environment that allows the model to accumulate and integrate knowledge of sequential tasks gradually.
> 
> 让模型逐步积累并整合来自连续任务的知识
> [!PDF|important] [[Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning.pdf#page=7&selection=499,0,546,3&color=important|Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning, p.7]]
> > The first is to select hard samples that are more difficult to classify by combining the maximum uncertainty and entropy maximization sampling strategies. 
> 
> 策略1：选择难以分类的困难样本
> 细节：结合“最大不确定性采样”和“熵最大化采样”实现
> 根据样本$x_{j}^i$预测概率分布中最高概率和次高概率之间的差值$△p_{x^{i}_{j}}$赖选择样本，差值越小表示模型对该样本分类的置信度越低，也就是模型不确定这个样本如何分类，因此会被归为困难样本
![[Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning.pdf#page=7&rect=36,44,151,68&color=important|Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning, p.7]]

## 熵最大化采样
通常会选择具有高熵值的样本$E(x_{j}^i)$的样本，熵值越高表示该模型对该样本的预测约不稳定
> [!PDF|important] [[Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning.pdf#page=7&selection=853,0,919,6&color=important|Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning, p.7]]
> > The entropy maximization sampling usually selects a sample with a relatively high entropy value 𝐸 ( 𝑥𝑖 𝑗 ) . A higher entropy value indicates that the model has lower confidence in the sample

![[Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning.pdf#page=7&rect=303,406,419,438&color=important|Continual face forgery detection based on relation-aware spatial-frequency interaction aggregation and contrastive learning, p.7]]


一般通过熵最大化采样和样本回放机制的筛选出来的样本都是模型最容易忽略或者遗忘分类不出来的样本，用这些样本进行回放再次训练就可以帮助模型更好的巩固对旧任务知识的记忆

# 关于实验

这篇论文的核心是解决模型在学习新的人脸伪造技术时，会“忘记”旧技术的问题（即灾难性遗忘）。其提出的框架结合了**多视角知识蒸馏**和**混合采样回放机制**。

---

### 一、 实验设置

#### 1. 数据集

论文在四个公开数据集上进行了评估：

- **FaceForensics++ (FF++)**: 包含4种伪造技术（Deepfakes-DF, Face2Face-F2F, FaceSwap-FS, NeuralTextures-NT）和真实视频。使用C23（高质量压缩）版本。
- **Deepfake Detection (DFD)**
- **Celeb-DF v2 (CDF2)**
- **DeepFake Detection Challenge-Preview (DFDC-P)**

#### 2. 持续学习场景（评估协议）

论文设计了两种主要的持续学习场景来评估模型：

**a) 数据集增量学习**

- **任务序列**：模型按顺序学习来自不同数据集的伪造技术。
- **示例序列**：`FF++` → `DFDC-P` → `CDF2`
- **目标**：评估模型在接触新数据集后，能否保持对旧数据集的检测能力。

**b) 伪造类型增量学习**

- **任务序列**：模型在FF++数据集内部，按顺序学习不同的伪造技术。
- **示例序列**：`DF` → `F2F` → `FS` → `NT`
- **目标**：评估模型在学会识别一种新的伪造方法（如FaceSwap）后，是否还能识别之前学过的伪造方法（如Deepfakes）。

#### 3. 对比方法

论文将提出的CFFD框架与多种基线方法进行比较：

- **FT**：直接在新任务数据上微调模型。这是性能下限，会产生严重的遗忘。
- **基于正则化的方法**：
    - **LwF**: 通过知识蒸馏保留旧任务的输出。
    - **EWC**: 通过计算参数的重要性，限制重要参数在学新任务时变化过大。
- **基于回放的方法**：
    - **ER**: 随机保存一部分旧任务的数据，与新任务数据一起训练。
    - **iCaRL**: 选择对每个类最有代表性的样本进行回放。
- **其他前沿的持续深度学习检测方法**：
    - **CoReD**, **HDU** 等。

---

### 二、 实验结果与效果分析

#### 1. 主要实验结果（伪造类型增量学习）

下表展示了在 `DF → F2F → FS → NT` 序列上的性能。关键指标是：

- **ACC/AUC per Task**: 学完所有任务后，在每个任务测试集上的性能。
- **平均性能**: 所有任务性能的平均值。
- **抗遗忘性能**: 模型在旧任务上的平均性能，值越高说明遗忘越少。

|方法|Replay Size|DF (ACC/AUC)|F2F (ACC/AUC)|FS (ACC/AUC)|NT (ACC/AUC)|平均ACC|抗遗忘ACC|
|---|---|---|---|---|---|---|---|
|**FT (下限)**|0|0.9979|0.8810|0.7946|0.7337|0.8518|0.7423|
|**LwF**|0|0.9979|0.8617|0.8315|0.7433|0.8586|0.8142|
|**EWC**|0|0.9997|0.9351|0.7816|0.6075|0.8310|0.7747|
|**ER**|500|0.9970|0.9317|0.8793|0.7433|**0.8878**|0.8510|
|**CoReD**|500|0.9911|0.9612|0.8606|0.8097|0.9057|0.8710|
|**HDU**|500|0.9997|0.9663|0.9516|0.8890|0.9517|0.9381|
|**CFFD (Ours)**|**500**|**0.9997**|**0.9663**|**0.9516**|**0.9137**|**0.9579**|**0.9381**|

**分析结论**：

1. **严重遗忘**：`FT`方法在学完所有任务后，在第一个任务（DF）上的性能从99.79%暴跌至73.37%，证明灾难性遗忘确实存在。
2. **CFFD有效性**：提出的`CFFD`框架在**平均性能**和**抗遗忘性能**上均达到最佳。它不仅对新任务（NT）的检测能力最强，对旧任务（如DF）的性能保持也最好（99.97%）。
3. **回放机制的重要性**：使用回放样本的方法（ER, CoReD, HDU, CFFD）普遍优于不使用回放的方法（LwF, EWC），说明保留部分旧数据至关重要。
4. **CFFD优势**：CFFD通过**多视角知识蒸馏**和**智能的混合采样回放**，比简单的回放方法（ER）和其他的持续深度学习检测方法（CoReD, HDU）效果更好，实现了更优的知识保留和迁移。

#### 2. 定性分析（可视化）

论文还提供了注意力图（Grad-CAM）来直观展示模型关注区域。

- **CFFD模型**的注意力更集中在伪造痕迹明显的区域，如嘴巴周围、面部轮廓等。
- 对于不同的伪造类型，关注区域也不同，例如：
    - **DF/F2F/FS**：关注整个面部区域。
    - **NT**：更集中地关注嘴巴区域（因为NT只修改嘴巴）。
- 这证明了CFFD框架能够学习到更具判别性和泛化性的特征。

---

### 三、 总结

该论文通过严谨的实验设计（两种增量学习场景、与多种基线对比）和全面的评估指标（ACC, AUC, 抗遗忘性能），有力地证明了其提出的CFFD框架在**持续人脸伪造检测**任务上的有效性。

**核心优势**：

- **有效缓解遗忘**：通过知识蒸馏和回放机制，在学新任务时能很好地保留旧知识。
- **泛化能力强**：在多个数据集和伪造类型上都表现出色。
- **实用性强**：为在实际应用中部署能够持续进化、适应新伪造技术的检测模型提供了可行的解决方案。

