> [!PDF|有用的信息] [[Forecasting directional movements of stock prices for intraday trading using LSTM and random forests.pdf#page=1&selection=27,0,47,92&color=有用的信息|Forecasting directional movements of stock prices for intraday trading using LSTM and random forests, p.1]]
> > We employ both random forests and LSTM networks (more precisely CuDNNLSTM) as training methodologies to analyze their effectiveness in forecasting out-of-sample directional movements of constituent stocks of the S&P 500 from January 1993 till December 2018 for intraday trading. We introduce a multi-feature setting consisting not only of the returns with respect to the closing prices, but also with respect to the opening prices and intraday returns. As trading strategy, we use Krauss et al. (2017) and Fischer & Krauss (2018) as benchmark. On each trading day, we buy the 10 stocks with the highest probability and sell short the 10 stocks with the lowest probability to outperform the market in terms of intraday returns – all with equal monetary weight. Our empirical results show that the multi-feature setting provides a daily return, prior to transaction costs, of 0.64% using LSTM networks, and 0.54% using random forests. Hence we outperform the single-feature setting in Fischer & Krauss (2018) and Krauss et al. (2017) consisting only of the daily returns with respect to the closing prices, having corresponding daily returns of 0.41% and of 0.39% with respect to LSTM and random forests, respectively.1 Keywords: Random forest, LSTM, Forecasting, Statistical Arbitrage, Machine learning, Intraday trading
> 
> 
我们采用随机森林和LSTM网络（更准确地说是CuDNNLSTM）作为训练方法，分析其在1993年1月至2018年12月期间对标准普尔500指数成分股日内交易方向性走势进行样本外预测的有效性。我们引入了一种多特征设置，不仅包含基于收盘价的收益率，还包含基于开盘价的收益率以及日内收益率。在交易策略方面，我们以Krauss等（2017）以及Fischer与Krauss（2018）的研究作为基准。在每个交易日，**我们买入10只预测日内收益率最有可能跑赢市场的股票，同时卖空10只预测最不可能跑赢市场的股票——所有股票均以相等的资金权重进行交易**。我们的实证结果表明，多特征设置下，使用LSTM网络可实现每日0.64%的收益率（未扣除交易成本），而使用随机森林则为0.54%。因此，我们优于Fischer与Krauss（2018）以及Krauss等（2017）研究中仅基于收盘价日收益率的单特征设置，其对应的LSTM和随机森林每日收益率分别为0.41%和0.39%。

关键词：随机森林，LSTM，预测，统计套利，机器学习，==日内交易==
注意这里是日内交易，我们的研究是长期
似乎评估方式可以参照他们的评估，也就是用实际进行交易
> [!PDF|important] [[Forecasting directional movements of stock prices for intraday trading using LSTM and random forests.pdf#page=2&selection=26,41,27,13&color=important|Forecasting directional movements of stock prices for intraday trading using LSTM and random forests, p.2]]
> > As data set we use all stocks of the S&P 500 from the period of January 1990 until December 2018
> 
> 使用了标普500的数据
> [!PDF|important] [[Forecasting directional movements of stock prices for intraday trading using LSTM and random forests.pdf#page=3&selection=7,0,11,37&color=important|Forecasting directional movements of stock prices for intraday trading using LSTM and random forests, p.3]]]
> > Our methodology is composed of five steps. In the first step, we divide our raw data into study periods, where each study period is divided into a training part (for in-sample trading), and a trading part (for out-of-sample predictions). In the second step, we introduce our features, whereas in the third step we set up our targets. In the forth step, we define the setup of our two machine learning methods we employ, namely random forest and CuDNNLSTM. Finally, in the fifth step, we establish a trading strategy for the trading part
> 
> 第一步，将原始数据划分为若干个时间窗口
> 第二部，将时间窗口分为训练部分（样本内交易）和交易部分（样本外预测）
> 第三步，引入特征
> 第四步，设定采用的两种机器学习方法配置，随机森林+CuDNN-LSTM
> 第五步，建立一个交易策略
> ![[Forecasting directional movements of stock prices for intraday trading using LSTM and random forests.pdf#page=3&rect=130,314,501,474&color=important|Forecasting directional movements of stock prices for intraday trading using LSTM and random forests, p.3]]

[[金融研究思考]]

本质上这个研究也是使用时间窗口，并在时间窗口之内分出训练集和测试集，但是似乎每个时间窗口之间没有关系，是否可以将每个时间窗口进行适当的重合或者处理，将以前的数据包含进去呢？
> [!PDF|有用的信息] [[Forecasting directional movements of stock prices for intraday trading using LSTM and random forests.pdf#page=4&selection=2,1,120,2&color=有用的信息|Forecasting directional movements of stock prices for intraday trading using LSTM and random forests, p.4]]
> > or any stock s ∈ S and any time t ∈ {241, 242, . . . , Tstudy }, the feature set we provide to the random forest comprises of the following three signals: 1. Intraday returns: ir(s) t,m := cp(s) t−m op(s) t−m − 1, 2. Returns with respect to last closing price: cr(s) t,m := cp(s) t−1 cp(s) t−1−m − 1, 3. Returns with respect to opening price: or(s) t,m := op(s) t cp(s) t−m − 1,
> 
> 对于日内的收益率制定了相对于前一日、现对于开盘价的收益率，但是实际上没有关注金融学上的波动特征原理

