---
title: "LayerNorm vs RMSNorm"
description: "Comparison of Layer Normalization and RMS Normalization"
pubDate: 2026-04-01
topic: "Algorithm Notes"
---

# LayerNorm v.s. RMSNorm
**LayerNorm** 在 Attention 结构中，对计算得到的 Attention 矩阵进行 row-wise 的层归一化，其目的将 token 的“注意力embedding” re-centering 和 re-scaling，保证不同的token 处在同一个 embedding 空间中。
具体计算过程是
1. 针对 token $\mathbf{x}$ 的 embedding 向量$[x_1, ..., x_{d_{model}}]$ 计算均值和方差
$$\mu(\mathbf{x}) = \frac{1}{d_{model}}\sum_{i=1}^{d_{model}} x_i,\,\, \sigma^2(\mathbf{x}) = \frac{1}{d_{model}-1}\sum_{i=1}^{d_{model}} (x_i-\mu)^2$$
2. 对embedding 向量的每个元素进行归一化 $\bar{x}_i=\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}$
3. 仿射变换 $y_i = \bar{x}_i\cdot\gamma_i+\beta_i$，最后得到归一化的向量 $\mathbf{y}=[y_1,...,y_{d_{model}}]$

**RMSNorm** 的核心思想是：“重中心化”不是必须的，它只保留“重缩放”部分。
具体计算过程是
1. 针对 token $\mathbf{x}$ 的 embedding 向量$[x_1, ..., x_{d_{model}}]$ 计算均方根
$$\text{RMS}(\mathbf{x})=\sqrt{\frac{1}{d_{model}}\sum_{i=1}^{d_{model}}x_i^2 + \epsilon}$$
2. 对embedding 向量的每个元素进行归一化 $\bar{x}_i=\frac{x_i}{\text{RMS}(\mathbf{x})}$
3. 仿射变换 $y_i = \bar{x}_i\cdot g_i$，通常只使用增益参数$g_i$，最后得到归一化的向量 $\mathbf{y}=[y_1,...,y_{d_{model}}]$

**为什么要用RMSNorm替换LayerNorm？**
计算效率更高，防止梯度爆炸
实验证明，LayerNorm能够保持token的注意力embedding在同一空间的根本原因不在于抵消均值偏移，而是重缩放保持尺度一致。RMSNorm方法越过计算均值的方式，使用square sum 直接缩放，将token embedding统一在一个固定半径的“超球面”空间中。
这个方法去掉冗余，提高了计算效率，同时还能防止梯度爆炸。

# BatchNorm v.s. LayerNorm
BatchNorm 的核心逻辑是“跨样本归一化”。它在训练过程中，针对每一个 mini-batch中的所有样本，计算每一个特征维度的均值和方差。
BatchNorm 在处理NLP任务和Transformer结构的时候有很多缺陷。
1. 对 Batch size 严重依赖：BatchNorm 的效果取决于 Batch 统计量的准确性。如果 Batch Size 太小，计算出的均值和方差波动剧烈，会导致训练极其不稳定
2. 无法处理变长序列：自然语言的长度不易，在同一个 Batch 中，长句子和带 Padding 的短句子混在一起计算均值，会导致统计特征被污染
3. 训练与推理的不一致性：推理时使用的BatchNorm参数是训练时累积的移动平均统计量，而推理时构建的Batch和训练时构建的Batch会有比较大的迁移，这种差异会导致模型效果大幅下降。

# Pre-Norm v.s. Post-Norm
**Post-norm** (Transformer, BERT)
先进行子层计算，加到残差上，最后进行 LayerNorm。
$$x_{l+1} = \text{LayerNorm}(x_l + \text{Sublayer}(x_l))$$
**Pre-norm** (目前LLM模型的统一选择)
在进入子层计算之前，先对输入进行 LayerNorm。
$$x_{l+1} = x_l + \text{Sublayer}(\text{LayerNorm}(x_l))$$

两者的核心差异在于**训练稳定性**和**模型表达能力**的权衡。
在pre-norm中，我们展开递推公式得到：
$$
\begin{align}
x_1 & = x_0 + f_0(\text{LN}(x_0))\\
x_2 & = x_1 + f_1(\text{LN}(x_1)) = x_0 + f_0(\text{LN}(x_0)) + f_1(\text{LN}(x_1))\\
\cdots \\
x_L&  = x_L + f_L(\text{LN}(x_L)) = x_0 + f_0(\text{LN}(x_0)) + f_1(\text{LN}(x_1)) + ... + f_{L-1}(\text{LN}(x_{L-1}))\\
 & = x_0 + \sum_{l=0}^{N-1}f_{l}(\text{LN}(x_{l}))
\end{align}
$$
在反向传播计算梯度时，由于 $x_L$ 是各曾输出的直接累加，梯度中始终包含一个恒等项：
$$
\begin{align}
\frac{\partial \mathcal L}{\partial x_l} & = \frac{\partial \mathcal L}{\partial x_L}\cdot\frac{\partial x_L}{\partial x_l}\\
& = \frac{\partial \mathcal L}{\partial x_L}\cdot\frac{\partial }{\partial x_l}\left(x_l + \sum_{i=l}^{L-1}f_{i}(\text{LN}(x_{i}))\right)\\
& = \frac{\partial \mathcal L}{\partial x_L}\cdot\left(I + \frac{\partial }{\partial x_l}  \sum_{i=l}^{L-1}f_i(\text{LN}(x_i))\right)
\end{align}
$$
只要有 $I$ 存在，无论其他项有多小，都能保证梯度依然通过 $I$ 传递到第 $l$ 层，这样可以避免梯度消失，保证了训练的稳定性。
对比之下，post-norm因为LN函数套在残差外，失去了这个不变项，导致训练不稳定。残差的本意是为了给前面的层添加一个快速通道，保障梯度稳定回传，而post-norm削弱了这个快速通道，残差名存实亡，容易导致梯度消失，难以训练。相反，由于pre-norm完整保留了残差，导致可学习的空间变小，因此训练效果的上限不如post-norm。

**learning rate warm-up** 是指学习率随着轮数逐渐增长到目标学习率。
Post-norm方法需要学习率warm-up是为了解决初始化阶段的“梯度信噪比”失衡。具体来讲，在训练初期，模型权重是随机初始化的，输出完全是噪声；并且因为完全随机导致靠近输出层的梯度值很大（因为链式法则）。如果直接使用目标学习率，那么 $\Delta W=\eta\cdot W$ 会剧烈抖动，巨大的更新可能会将权重推向一个饱和区，导致模型迅速发散。
Warm-up 方法在初始阶段将 $\eta$ 设置地非常小，强制模型以微笑的步长缓慢移动。在这个过程中，LN的参数 $(\gamma,\beta)$ 和 Attention的权重会先进行对齐，当信号稳定之后再增加学习率可以避免发散。

**DeepNorm** 
$$
x_{l+1} = \text{LayerNorm}(\alpha\cdot x_l + G_l(x_l,\theta_l))
$$
同时，它会对残差分支内的参数 $\theta_l$ 进行缩放，缩放系数为 $\beta$。
$\alpha$ 和 $\beta$ 这两个参数不是学习出来的，而是根据模型的层数提前计算好的常量。可以这样理解，DeepNorm方法通过针对每一层使用不同但固定的 $\alpha$ 和 $\beta$ ，从架构层面限制网络每一层对最终输出的影响力，预先抵消深度增加带来的梯度不稳定性。
