
- [x] 通读人脸篡改检测的开山之作（2015-2020关键论文速览）
- [x] 整理现有公开人脸篡改数据集列表（FF++, Celeb-DF-V2, DFDC等）
- [x] 复现并跑通1-2个经典baseline方法（Xception-Fake/ Mesonet4）
- [x] 搞清楚评价指标：AUC, EER, ACC分别代表什么
- [x] 对比不同模态（RGB/频域/噪声）对检测效果的影响
- [x] 研究当前SOTA都在用什么trick（attention/transformer/self-ensemble）
- [x] 找2篇最新顶会论文（CVPR/ICCV/ECCV）做精读笔记
- [x] 跑通PyTorch版人脸检测器+后处理demo
- [x] 列出自己工作可能的创新点或待解决难点

# 2025年11月23日
- [x] **仔细回顾** 咱们的防伪任务特点，思考它对全局信息和局部信息的侧重。
- [ ] **执行可视化实验**：
    - [ ] 尝试可视化ViT和当前LocalAttention2D的注意力权重，对比它们在图片上关注的区域。
    - [ ] 记录观察结果。
- [x] **探索混合注意力策略**：
    - [x] 尝试构建一个**混合模型**，例如在模型的**前半部分**使用`LocalAttention2D`，**后半部分**使用`AttentionBlock`。
    - [x] 或者尝试**交替使用**这两种注意力模块。
- [ ] **研究并考虑实现窗口间交互机制**：
    - [ ] 了解Swin Transformer的`shifted window attention`原理。
    - [ ] [教程](https://www.bilibili.com/video/BV1GPymBwEu1?t=462.7&p=21)
    - [ ] [另外一个教程](https://www.bilibili.com/video/BV13L4y1475U?t=4.6)
    - [ ] 尝试在`LocalAttention2D`的基础上，实现简单的窗口偏移来增强信息交流。
- [x] **调整训练策略**：
    - [x] 如果新的模型效果不佳，尝试增加训练`epochs`。
    - [x] 检查学习率调度器是否合适。cosine

# 2025年11月24日
- [ ] 将现有的ViT特征提取改成swan-T
- [ ] 给现有的Rine双分支进行消融实验
- [ ] 将rine相关的模块都进行可视化实验
- [ ] 思考是否能将SSCA相关的注意力使用SwinTransformer进行优化