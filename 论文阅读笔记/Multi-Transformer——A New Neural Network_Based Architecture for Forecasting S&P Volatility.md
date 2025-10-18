# 摘要内容

> [!PDF|important] [[Multi-Transformer A New Neural Network-Based Architecture for Forecasting S&P Volatility.pdf#page=1&selection=140,0,146,55&color=important|Multi-Transformer A New Neural Network-Based Architecture for Forecasting S&P Volatility, p.1]]
> > GARCH-based models [ 3,4 ] are widely used for volatility forecasting purposes. 
> 使用模型：基于==GARCH==的模型
> 考虑了波动率聚类现象

## GARCH模型

GARCH 模型是 “Generalized Autoregressive Conditional Heteroskedasticity”（广义自回归条件异方差模型）的缩写，是时间序列分析中用于捕捉**波动率聚类**（Volatility Clustering）现象的核心模型。它由 Bollerslev 在 1986 年基于 Engle（1982）的 ARCH 模型（自回归条件异方差模型）扩展而来，解决了 ARCH 模型对 “滞后项数量依赖过多” 的缺陷，成为金融、经济等领域分析波动率的基础工具。

## 一、为什么需要 GARCH 模型？—— 核心应用场景

在时间序列数据（尤其是金融数据，如股票收益率、汇率波动）中，存在一个典型特征：**大的波动往往集中在一段时间内，小的波动也会集中在另一段时间内**（即 “波动率聚类”）。例如，股市暴跌后常伴随持续的剧烈震荡，而平稳期则波动较小。

传统的时间序列模型（如 ARMA 模型）假设 “方差恒定”（同方差），无法解释这种 “方差随时间变化”（异方差）的现象，而 GARCH 模型的核心价值就是通过 “条件异方差” 的设定，精准刻画波动率的动态变化。

## 二、GARCH 模型的核心逻辑与数学形式

GARCH 模型的本质是 “用过去的波动率信息预测未来的条件波动率”，分为**均值方程**和**条件方差方程**两部分，核心是条件方差方程。

### 1. 基本假设

- 时间序列的**无条件方差恒定**（长期均值稳定），但**条件方差（给定过去信息的方差）随时间变化**；
- 条件方差仅依赖于 “过去的误差平方项” 和 “过去的条件方差”，即波动率具有 “自相关性”。

### 2. 标准 GARCH (p,q) 模型的数学表达

“GARCH (p,q)” 中，`p`表示 “滞后的条件方差项数量”，`q`表示 “滞后的误差平方项数量”，最常用的是**GARCH(1,1)**（仅需 1 个滞后方差项和 1 个滞后误差项，即可解释大部分波动率聚类现象）。

#### （1）均值方程

首先刻画序列本身的均值趋势，通常假设序列围绕常数均值波动（或结合 ARMA 模型扩展）：$y_t = \mu + \varepsilon_t$ 其中：

- \($y_t$\)：第 t 期的时间序列值（如股票收益率）；
- ($\mu$)：序列的无条件均值（长期平均水平）；
- \($varepsilon_t$)：第 t 期的误差项（扰动项），且满足 \($\varepsilon_t = \sigma_t \cdot z_t$\)（\(z_t\) 是独立同分布的随机变量，通常假设为**标准正态分布**或**t 分布**，均值为 0、方差为 1）。

#### （2）条件方差方程（核心）

这是 GARCH 模型的关键，定义 “第 t 期的条件方差 \(\sigma_t^2\)” 如何由历史信息决定： $$\sigma_t^2 = \omega + \alpha_1 \varepsilon_{t-1}^2 + \alpha_2 \varepsilon_{t-2}^2 + ... + \alpha_q \varepsilon_{t-q}^2 + \beta_1 \sigma_{t-1}^2 + \beta_2 \sigma_{t-2}^2 + ... + \beta_p \sigma_{t-p}^2$$

对最常用的**GARCH(1,1)**，方程简化为： $$\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

各参数的含义与约束：

- \($\omega$\)：常数项，需满足 \(\omega > 0\)（保证条件方差为正）；
- \($\alpha$\)：“滞后误差平方项的系数”，反映 “过去冲击对当前波动率的影响”（\($\alpha > 0$\)，冲击越大，当前波动率越高）；
- \($\beta$\)：“滞后条件方差项的系数”，反映 “波动率的持续性”（\($\beta > 0$\)，\($\beta$\) 越接近 1，波动率的持续时间越长）；
- 约束条件：\($\alpha + \beta < 1$\)（保证无条件方差存在且稳定，若 \($\alpha + \beta = 1$\)，则退化为 IGARCH 模型，波动率具有无限持续性）。

## 三、GARCH 模型的扩展形式

标准 GARCH (1,1) 虽应用广泛，但无法解决部分特殊场景（如 “杠杆效应”），因此衍生出多个扩展模型：

| 模型类型                          | 核心改进                                                                                                                                                                           | 适用场景                                          |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------- |
| **EGARCH 模型**（指数 GARCH）       | 1. 用对数形式刻画条件方差，自动保证方差非负； 2. 允许 “正负冲击对波动率的影响不对称”（杠杆效应）                                                                                                                          | 金融市场中，“负收益冲击（如股价下跌）对波动率的影响大于正收益冲击” 的场景（如股票市场） |
| **TGARCH 模型**（门限 GARCH）       | 引入 “虚拟变量” 区分正负冲击，直接刻画不对称性： $\sigma_t^2 = \omega + (\alpha + \gamma I_{t-1}) \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2$（$I_{t-1}$\) 为虚拟变量，\($\varepsilon_{t-1}<0$\)时取 1，否则取 0） | 需明确区分 “正负冲击影响差异” 的场景，计算比 EGARCH 更简单           |
| **GARCH-M 模型**（GARCH-in-Mean） | 将 “条件方差 / 标准差” 引入均值方程，刻画 “波动率与收益的关系”： \($y_t = \mu + \delta \sigma_t + \varepsilon_t$\)（\($\delta$\) 为 “风险溢价系数”）                                                               | 分析 “风险与收益正相关” 的场景（如资产定价：波动率越高，预期收益越高）         |
| **IGARCH 模型**（积分 GARCH）       | 放松 “\($\alpha + \beta < 1$\)” 的约束，允许 \($\alpha + \beta = 1$\)，此时波动率具有 “无限持续性”                                                                                                  | 刻画长期波动趋势（如恶性通胀时期的物价波动、金融危机后的持续震荡）             |

## 四、GARCH 模型的建模步骤

使用 GARCH 模型分析数据需遵循严格的流程，以确保结果可靠：

1. **数据预处理**
    - 选择合适的时间序列（通常为高频数据，如日度 / 小时度收益率）；
    - 检验序列的平稳性（如 ADF 检验）：若序列非平稳，需先差分（如股票价格→股票收益率）；
    - 检验序列的 “异方差性”（如 ARCH-LM 检验）：若存在显著异方差，再使用 GARCH 模型。
2. **模型识别与选择**
    - 通过 “ACF/PACF 图” 或 “信息准则（AIC、BIC）” 确定 GARCH (p,q) 的阶数（p,q），通常从 GARCH (1,1) 开始尝试；
    - 若存在杠杆效应，进一步选择 EGARCH 或 TGARCH；若需分析风险溢价，选择 GARCH-M。

3. **参数估计**
    
    - 采用 “极大似然估计（MLE）” 估计模型参数（\(\omega, \alpha, \beta\) 等），需指定误差项 \(z_t\) 的分布（正态分布、t 分布或 GED 分布，t 分布更适合捕捉金融数据的 “尖峰厚尾” 特征）。
4. **模型诊断**
    
    - 检验 “残差的异方差性”：若残差无显著异方差（ARCH-LM 检验 p 值 > 0.05），说明模型已捕捉到波动率聚类；
    - 检验 “残差的正态性”：若残差非正态，可调整 \(z_t\) 的分布（如改用 t 分布）。
5. **波动率预测与应用**
    
    - 基于估计的模型预测未来的条件波动率（如预测未来 10 天的股市波动）；
    - 应用于风险控制（如 VaR 计算）、资产定价、交易策略设计等场景。

## 五、GARCH 模型的优缺点

### 优点

1. **精准刻画波动率聚类**：解决了传统模型 “同方差假设” 的缺陷，贴合金融数据的实际特征；
2. **灵活性强**：通过扩展模型（EGARCH、TGARCH 等）可处理杠杆效应、风险溢价等复杂问题；
3. **实用性高**：广泛应用于金融、经济领域，是波动率预测和风险分析的核心工具。

### 缺点

1. **对极端值敏感**：若数据中存在异常值（如黑天鹅事件），可能导致参数估计偏差；
2. **假设条件较严格**：需满足 “误差项独立同分布”“参数约束（\(\alpha + \beta < 1\)）” 等，实际数据可能不满足；
3. **仅关注波动率，忽略其他信息**：无法结合宏观经济指标、政策变化等外部因素对波动率的影响。

## 六、总结

GARCH 模型是时间序列分析中 “异方差建模” 的里程碑，其核心是通过 “过去的冲击和波动率” 预测未来的条件波动率，从而解决了 “波动率聚类” 这一关键问题。从标准 GARCH (1,1) 到 EGARCH、TGARCH 等扩展模型，它不断适应复杂的实际场景，成为金融领域（如股市波动预测、VaR 计算）、经济领域（如通胀波动分析）不可或缺的工具。

在实际应用中，需结合数据特征选择合适的模型类型，严格遵循 “预处理→识别→估计→诊断” 的建模流程，才能充分发挥 GARCH 模型的价值。


> [!PDF|important] [[Multi-Transformer A New Neural Network-Based Architecture for Forecasting S&P Volatility.pdf#page=1&selection=147,40,148,22&color=important|Multi-Transformer A New Neural Network-Based Architecture for Forecasting S&P Volatility, p.1]]
> > because it takes into consideration the volatility clustering observed by
> 
> 股票序列数据往往有波动聚类现象
# 波动聚类现象
**高波动聚类**：比如A股市场，在2020年新冠疫情爆发初期，22年美联储激进加息周期，故事连续多日出现了2-3%的答复涨跌，甚至出发了熔断，这一阶段的故事波动集中出现，形成了高波动簇。

**低波动聚类**：2021年Q2-3的A股消费版块，2023上半年的港股蓝筹股，多数时间单日涨跌幅小于1%，连续数周波动维持在窄区间，这一阶段的小波动集中出现，形成了低波动簇

高波动簇和低波动簇分别对应了市场的振荡期和平稳期

**汇率市场受到政策冲击之后的持续波动**
2024年美联储江西预期落地，美元兑人民币汇率在消息公布之后当日波动幅度扩大到500点（日常200点波动）。随后一周内仍然维持400-500的高波动，意味着*政策冲击引发的大波动并非单日就会消失，而是持续聚集数周*

## 波动聚类现象的核心特征
1. **时间聚集性**，呈现阶段性，一段时间低，一段时间高，
	1. 可以绘制收益率的绝对、平方值时间序列图（波动常用收益率的绝对、平方值代理）可以见到明显的高低波动快
2. **波动率自相关性**，档期波动与过去几期的波动有显著相关，也就是今天如果波动比较大，那么明天波动大的概率就就会比较高
	1. 对收益率的绝对、平方值进行自相关性分析（ACF、PACF）之后多期的自相关系数仍然显著为正
3. **尖峰厚尾**，存在波动聚类的金融数据，其收益率分布通常不是正态分布，而是尖峰——均值附近的概率密度更高，厚尾——极端涨跌的概率高于正态分布，也就是极端波动的聚类导致了厚尾
	1. 使用Jarque-Bera正态性检验（P<0.05拒绝正态分布假设）绘制QQ图可见尾部屁哪里45度线验证

## 出现这个现象的原因
信息冲击的特性+市场参与者的行为模型共同决定，也就是信息冲击的“聚集性”和“传导性”
除了信息冲击，市场参与者的行为模式（尤其是 “羊群效应”）会放大波动聚类：
- 当市场出现首次大波动（如某股票因利空下跌 5%），部分投资者会因 “恐慌情绪” 跟风卖出，导致次日继续下跌 3%-4%（波动聚集）；
- 当市场长期平稳（如指数连续 10 周涨跌幅 <1%），投资者会因 “风险感知降低” 减少交易，市场流动性下降，波动进一步维持低位（波动聚集）。
> [!PDF|有用的信息] [[Multi-Transformer A New Neural Network-Based Architecture for Forecasting S&P Volatility.pdf#page=2&selection=138,0,157,20&color=有用的信息|Multi-Transformer A New Neural Network-Based Architecture for Forecasting S&P Volatility, p.2]]
> > The aim of this paper is to introduce a more accurate stock volatility model based on an innovative machine and deep learning technique. For this purpose, hybrid models based on merging Transformer and Multi-Transformer layers with other approaches such as GARCH-based algorithms or LSTM units are introduced by this paper. Multi-Transformer layers, which are also introduced in this paper, are based on the Transformer architecture developed by [54 ]. Transformer layers have been successfully implemented in the field of natural language processing (NLP). Indeed, the models developed by [55 , 56] demonstrated that Transformer layers are able to overcome the performance of traditional NLP models. Thus, this recently developed architecture is currently considered the state-of-the-art in the field of NLP. In contrast to LSTM, Transformer layers do not incorporate recurrence in their structure. 
> 
> 引入基于transformer和多transformer层与其他方法（基于GARCH算法或者LSTM单元）相结合的混合模型，并引入了多transformer层，去除了不需要的NLP词嵌入的等

# 论文使用方法
> [!PDF|有用的信息] [[Multi-Transformer A New Neural Network-Based Architecture for Forecasting S&P Volatility.pdf#page=3&selection=63,5,64,8&color=有用的信息|Multi-Transformer A New Neural Network-Based Architecture for Forecasting S&P Volatility, p.3]]
> > roposed architectures and benchmark models are fitted using the rolling window approach
> 
> 使用了滑动窗口
![[Multi-Transformer A New Neural Network-Based Architecture for Forecasting S&P Volatility.pdf#page=4&rect=149,580,564,747&color=有用的信息|Multi-Transformer A New Neural Network-Based Architecture for Forecasting S&P Volatility, p.4]]
然后对于每个窗口都求方差
![[Multi-Transformer A New Neural Network-Based Architecture for Forecasting S&P Volatility.pdf#page=4&rect=276,475,453,528&color=有用的信息|Multi-Transformer A New Neural Network-Based Architecture for Forecasting S&P Volatility, p.4]]
然后将时间序列的方差和原本的数据分别放进注意力机制头（他要干嘛）
Transformer 层和 LSTM 层能够处理时间序列，因此在拟合这些层时，会考虑先前变量最后 10 个观测值的滞后项

将对数收益率（原始数据）反映了资产价格的变化幅度与方向，收益率的标准差（波动率相关统计量）体现了这种变化的波动程度，多头注意力机制擅长处理序列数据并捕捉不同位置元素之间的依赖关系，将这两个数据结合输入可以让模型同时关注资产价格变化的水平和波动特性，从多维度挖掘数据中的模式，提高预测的准确度

 