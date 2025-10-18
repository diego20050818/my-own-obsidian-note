 > [!PDF|yellow] [[A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices.pdf#page=1&selection=66,0,82,31&color=yellow|A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices, p.1]]
> > In recent years, numerous methods for time series predictions have been proposed [2–13]. These methods can be classified into the following three categories: traditional econometric models, machine learning approaches and deep learning models. The autoregressive integrated moving average model (ARIMA) is a popular statistical model applied to time series prediction
> 
> 对原油期货的预测，传统计量经济模型、机器学习方法和深度学习模型。自回归积分滑动平均模型（ARIMA）是一种广泛应用于时间序列预测的概率模型。

## 1️⃣ ARIMA 模型的基本概念

ARIMA 是一种用于 **时间序列预测** 的统计模型，它结合了三种特性：
1. **自回归（AR, Autoregressive）**
    - 当前值与过去若干时刻的值线性相关。
    - 表达式：
        $$Xt=ϕ1Xt−1+ϕ2Xt−2+⋯+ϕpXt−p+ϵtX_t = \phi_1 X_{t-1} + \phi_2 X_{t-2} + \dots + \phi_p X_{t-p} + \epsilon_t$$
        - pp 是自回归阶数。    
        - ϵt$\epsilon_t$ 是白噪声。    
2. **差分积分（I, Integrated）**
    - 用于处理 **非平稳时间序列**，即数据存在趋势或季节性变化。
    - 通过对序列做 dd 次差分（Δ）将其转化为平稳序列：
        $$Yt=ΔdXtY_t = \Delta^d X_t$$
    - dd 是差分阶数。
        
3. **移动平均（MA, Moving Average）**
    - 当前值与 **过去若干随机误差项** 线性相关。
    - 表达式：
        $$Xt=ϵt+θ1ϵt−1+⋯+θqϵt−qX_t = \epsilon_t + \theta_1 \epsilon_{t-1} + \dots + \theta_q \epsilon_{t-q}$$
        - qq 是移动平均阶数。

---

## 2️⃣ ARIMA 模型的数学形式
将 AR、I、MA 结合起来得到 ARIMA(p,d,q) 模型：
$$ΔdXt=ϕ1ΔdXt−1+⋯+ϕpΔdXt−p+ϵt+θ1ϵt−1+⋯+θqϵt−q\Delta^d X_t = \phi_1 \Delta^d X_{t-1} + \dots + \phi_p \Delta^d X_{t-p} + \epsilon_t + \theta_1 \epsilon_{t-1} + \dots + \theta_q \epsilon_{t-q}$$

- 左边：经过dd 次差分后的平稳序列
- 右边：自回归部分 + 移动平均部分 + 白噪声

---

## 3️⃣ 模型求解的核心步骤

1. **平稳性检测**：ADF 检验、KPSS 检验等。
2. **差分处理**：如果序列非平稳，选择合适的 dd 做差分。
3. **确定 p 和 q**：
    - 使用 **ACF**（自相关函数）和 **PACF**（偏自相关函数）判断。
    - 或通过信息准则（AIC、BIC）自动选择。
4. **模型拟合**：估计参数 ϕi,θj$\phi_i$, $\theta_j$。
5. **模型检验**：残差是否为白噪声。
6. **预测**：对未来序列进行预测。

---
## 4️⃣ 应用场景
- **金融**：股票价格、利率预测
- **经济**：GDP、消费指数
- **工业**：负荷预测、产量预测
- **社会**：人口、疾病疫情趋势预测
---

## 以前的研究试过的方法汇总
### 机器学习

 > [!PDF|有用的信息] [[A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices.pdf#page=20&selection=198,4,203,11&color=有用的信息|A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices, p.20]]
> > W. Huang, Y. Nakamori, and S.-Y. Wang, “Forecasting stock market movement direction with support vector machine,” Computers & Operations Research, vol. 32, no. 10, pp. 2513– 2522, 2005.
> 
> 支持向量机+分类方法
> [!PDF|有用的信息] [[A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices.pdf#page=2&selection=1,28,4,14&color=有用的信息|A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices, p.2]]
> > xplored the forecasting ability of SVM for financial movement direction and proposed a combining model based on SVM and classification methods. Ghiassi et al.
> 
> 事件序列事件预测动态神经网络
> > [!PDF|有用的信息] [[A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices.pdf#page=2&selection=7,28,10,50&color=有用的信息|A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices, p.2]]
> > Liao and Wang [6] established an improved neural network, the stochastic time-effective neural network model, and analyzed the volatility statistics characteristics of the Chinese stock price indices
> 
> 随机时间有效神经网络模型
> 比ARIMA有更高的准确性
> > [!PDF|有用的信息] [[A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices.pdf#page=2&selection=11,10,14,59&color=有用的信息|A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices, p.2]]
> > stablished a hybrid model by combining the principle component analysis (PCA) algorithm and random time-effective neural networks (STNN) and explored the predictive performance by considering financial time series.
> 
> PCA+STNN随机时间有效神经网络

比较耗时，而且对大数据也没有很好，所以有人使用了深度学习的相关办法
### 深度学习方法
> [!PDF|有用的信息] [[A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices.pdf#page=2&selection=34,46,36,14&color=有用的信息|A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices, p.2]]
> > he transmission of historical information can be realized by recurrent neural networks (RNNs
> 
> 循环神经网络
> > [!PDF|有用的信息] [[A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices.pdf#page=2&selection=38,42,42,1&color=有用的信息|A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices, p.2]]
> > ultilayer perceptrons (MLP) 
> 
> 多头注意力机制
> > [!PDF|有用的信息] [[A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices.pdf#page=2&selection=45,0,52,8&color=有用的信息|A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices, p.2]]
> > Elman recurrent neural networks (ERNN) w

> Elman recurrent neural networks（ERNN）即 Elman 递归神经网络，是一种典型的递归神经网络（RNN），由杰弗里·埃尔曼（Jeff Elman）在1990年提出。以下是关于它的详细介绍：
> 
> ==网络结构 ==：Elman 神经网络主要由输入层、隐藏层、承接层和输出层组成。输入层接收外部输入的序列数据；隐藏层对输入层的数据进行非线性变换；承接层是 Elman 神经网络的独特之处，它将上一时刻隐藏层的输出作为当前时刻的输入，从而引入了时间信息，其神经元数量与隐藏层相同；输出层输出网络的预测结果。
> 
> ==工作原理== ：在每个时间步，输入向量进入输入层，然后传递到隐藏层。隐藏层神经元的输入不仅包含来自输入层的加权输入，还包含来自承接层的反馈信息，这些输入经过激活函数进行非线性变换后，传递到输出层，生成预测结果。
 

> [!PDF|important] [[A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices.pdf#page=2&selection=63,15,67,8&color=important|A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices, p.2]]
> > . Long short-term memory network (LSTM) is a type of deep learning method devised to deal with the longterm dependence problems for a special purpose [18]. The network structure of LSTM is much more complex than that of RNNs,
> 
> 长短记忆网络比传统的循环神经网络复杂度高，对这种时间序列更难呢个够处理长时间问题，发现数据的隐藏模式，而且能够有效解决传统循环神经网络的梯度爆炸和梯度消失问题

> [!PDF|有用的信息] [[A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices.pdf#page=2&selection=101,8,101,52&color=有用的信息|A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices, p.2]]
> >  different kinds of hybrid forecasting models
> 
> 不能只用一种模型进行预测，混合模型互补才是出路
> 
>  [!PDF|有用的信息] [[A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices.pdf#page=2&selection=103,36,105,10&color=有用的信息|A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices, p.2]]
> > he hybrid models based on decomposition and prediction have been widely recognized
> 
> 基于分解和预测的模型，由非线性分解模型和预测模型组合而成

> [!PDF|有用的信息] [[A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices.pdf#page=2&selection=144,26,149,18&color=有用的信息|A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices, p.2]]
> > . Wang and Wang [28] combined empirical mode decomposition (EMD) method with random time strength neural network to predict global stock indices, and the empirical results showed that the proposed approach veritably has a great effect in predicting stock market fluctuations
> 
> 经验模态分解方法和随机时间强度神经网络预测全球股票指数
> 这个方法适合用来描述股票市场的波动，但是似乎需要检查是短期波动还是长期市场变化
> 

> [!PDF|有用的信息] [[A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices.pdf#page=2&selection=149,20,165,6&color=有用的信息|A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices, p.2]]
> > Wang et al. [29] established a two-layer decomposition model and then developed an ensemble approach by integrating the fast ensemble empirical mode decomposition method (FEEMD), variational mode decomposition (VMD), and optimized backpropagation neural network by firefly algorithm (FA-BPNN). The empirical results indicated that the developed new model has exceptional forecasting implementation in electricity price series
> 
> 建立一个两层分解模型，通过整合快速集合经验模态分解法（FEEMD）、变分模态分解（VMD）以及经过萤火虫算法优化的反向传播神经网络（FA-BPNN）开发了一种集合算法，在电价序列预测方面表现出色

 > [!PDF|important] [[A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices.pdf#page=3&selection=5,34,7,14&color=important|A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices, p.3]]
> > The mechanism of SW is to measure historical information in conformity with the time of occurrence.
> 
> SW基于随机过程，符合真实的交易市场，也符合预测模型中的门控机制
> 按照历史信息的发生时间来衡量起价值，发生的时间越近，包含的数据信息对呈现未来信息的价值就越大，可以利用历史价格数据来捕捉能源期货序列中的波动统计特征
> > [!PDF|important] [[A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices.pdf#page=3&selection=25,26,29,22&color=important|A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices, p.3]]
> >  In addition, this research employs the WPD method to extract the original crude oil series for the first time and firstly improves the conventional LSTM model with stochastic time strength weights for the crude oil prices forecasting
> 
> 采用WPD方法对原始原油期货序列特征进行提取
> 利用随机时间强度权重对传统的LSTM模型进行改进用于预测
## WPD 小波包分解
基于小波变换的型号处理技术，可以将信号分解为不同频率的子带
学习链接：[(32 封私信  / 8 条消息) 小波包变换（Wavelet Packet Transform）的学习笔记 - 知乎](https://zhuanlan.zhihu.com/p/58596902)

# 论文创新点
## 模型创新点
1. 使用小波变换分解WPD将原始的价格序列分解为若干个不同频段的子序列（SSi）
2. 针对相应的子序列分别构建不同的滑动窗口长短记忆（SW-LSTM）模型
3. 最后整合预测结果，得到原始的能源期货序列的集成观测结果
## 模型评估创新点
 > [!PDF|important] [[A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices.pdf#page=3&selection=70,28,72,38&color=important|A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices, p.3]]
> > his research proposes a new error measurement method called multiorder multiscale complexity invariant distance (MMCID) 
> 
> 多阶多尺度复杂度不变距离

# 数据集
> [!PDF|important] [[A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices.pdf#page=3&selection=110,12,126,15&color=important|A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices, p.3]]
> > west Texas intermediate (WTI) futures prices series, Brent crude oil futures prices series, RBOB gasoline, and heating oil. These four datasets are from the New York Mercantile Exchange (NYMEX) energy futures market, which can be downloaded from https://www.wind.com.cn/. WTI crude oil price is widely applied in the pricing of US domestic crudes
> 
> WTI原油价格，包括西德克萨斯中质原油期货价格序列，布伦特原油期货价格序列，RBOB汽油期货价格序列和取暖油期货价格序列
> 下载链接：[纽约商品交易所](https://www.wind.com.cn/)

# 方法
小波变换是一种时频函数
![[A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices.pdf#page=3&rect=365,196,494,219&color=important|A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices, p.3]]
$W_{j,k}^n(t)$:小波包函数
- $j$:尺度参数，控制小包的伸缩（越大频率分辨率越高，时域越窄）
- $k$:平移参数，在时间轴上的平移
- $n$:调制、震荡参数，控制小波包的震荡模式（对应不同的频率细分，让小波包可以更加精细的分析信号高频成分
- $t$:时间变量

$2^{j/2}$:归一化因子，保证小波包函数在拨通尺度下能量守恒（就是保持伸缩后的总能量不变）

$2^jt-k$:伸缩与平移的复合变换，对t先做伸缩，让时域被压缩，频率放大，在做平移，调整小波包在时间轴的位置

![[A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices.pdf#page=3&rect=383,92,476,135&color=important|A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices, p.3]]


> [!PDF|不懂] [[A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices.pdf#page=3&selection=222,34,238,10&color=不懂|A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices, p.3]]
> > The first two wavelet packet functions are the scaling and mother wavelet functions:
> 
> 尺度函数和母小波函数之间就是关闭和开启小波包系数？
> n=0 和 n=1 本质上是**时间维度的区分**，系数的差异直接对应信号在这两个时刻的特征差异，体现了小波包分析 “**时频局部化**” 的核心能力 —— 既能观察频率特征，又能定位时间位置。
> 

总结
尺度函数（类似实线）,表现相对平滑，能量集中，可以大致反映出大尺度的趋势
$$n=1 \rightarrow W_{0,0}^1=ψ(t)$$
小波函数（类似虚线）呈现正负交替的震荡波形，展现细节
$$n=0 \rightarrow W_{0,0}^0=φ(t)$$

![[Pasted image 20250915174902.png]]
$$\omega_{j,k}^n=<f(t)W_{j,k}^n>=\int{f(t)W_{j,k}^ndt}$$
然后论文中将信号在频域上划分了不同的频段，然后在时域上用LSTM进行分析![[A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices.pdf#page=5&rect=45,463,559,734|A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices, p.5]]
$D$:细节系数，对应信号的高频，反应信号的细节和突变等等特征
$A$:近似系数，对应信号的低频，反应信号的大致趋势和整体轮廓等特征

首先对 “Actual data（实际数据）” 进行分解，得到不同的节点（如 D1、A1 等），然后不断对这些节点进一步分解，产生更多包含不同频率成分（由 D 和 A 组合体现不同频段）的子序列，最终到 $SS_1$到 $SS_8$ 这些分解后的子序列，每个子序列对应不同的频率范围，可用于后续的分析

# LSTM的部分
## LSTM模型架构
![[A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices.pdf#page=6&rect=73,557,537,739&color=important|A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices, p.6]]

> [!PDF|important] [[A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices.pdf#page=6&selection=15,18,18,25&color=important|A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices, p.6]]
> > prediction of financial market price series should integrate great amount of historical data, because the information represented in different periods has different impacts on future results
> 
> 金融市场价格序列的预测应该整合大量的历史数据，因为不同时期所呈现的信息对未来的结果影响不同，也就是说，距离离当前时间越近，那么该信息的影响越强烈
> SW-LSTM就是一个带有随机时间强度权重函数的长短期记忆网络

$$φ(t_{n}) = \frac{1}{\beta}\exp{
(\int_{t_{0}}^{t_{n}}\mu(t)dt+\int_{t_{0}}^{t_{n}}\omega(t)dB(t)})$$
$\beta$是市场深度参数，
$t_{0}$是数据集中最新时间点的时刻
$t_{n}$是数据集中的任意时间点
$B(t)$是标准布朗运动，通常认为是例子在液体中的随机运动
$\mu(t)$是漂移函数，主要指示趋势的变化
$\omega(t)$是波动函数，用来预测过程中对不确定事件的建模，其中
![[A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices.pdf#page=6&rect=326,402,566,468&color=important|A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices, p.6]]


模型的全局误差定义：

$$
E=
\frac{1}{N}\sum^N_{i=1}E_{t}=
\frac{1}{2N}\sum^N_{i=1}
\frac{1}{\beta}
\left( \int^{t_{n}}_{t{0}}\mu(t)dt+
\int \omega (t)dB(t)\right)
(d_{t_{n}}-y_{t_{n}})^2
$$
> [!PDF|有用的信息] [[A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices.pdf#page=6&selection=472,0,481,1&color=有用的信息|A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices, p.6]]
> > In the modelling process, based on the newly defined global error E, the model parameters are updated through the gradient descent method [10, 50, 51]. First, the partial derivative of each model parameter needs to be calculated from the global error function. Then, the principle of parameter update is as follows:
> 
> 这个玩意的梯度下降表示

# 算法步骤
![[A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices.pdf#page=8&rect=36,281,571,770&color=important|A New Hybrid Forecasting Model Based on SW-LSTM and Wavelet Packet Decomposition A Case Study of Oil Futures Prices, p.8]]
