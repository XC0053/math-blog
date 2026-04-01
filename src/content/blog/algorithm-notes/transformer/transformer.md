# Attention
### Self-Attention
![[Transformer_figure_01.png|140]]

Self-Attention是一种处理Seq2Seq任务的方法，它的优势是序列中的目标token能够获得整个序列的信息，同时还解决了长序列的信息衰减问题。

输入向量的序列 $x\in\mathbb{R}^{n\times d_{model}}$ 分别经过三个权重矩阵，映射为Query、Key、Value矩阵（$d_{model}$ 表示token嵌入向量空间的向量维度, $n$ 表示序列长度）
+ $Q= x W^Q$ 代表当前token“想要寻找什么”
+ $K= x W^K$ 代表序列中每个token分别“提供什么信息”
+ $V=x W^V$ 代表当前token“包含的具体内容”
其中 $W^Q, W^K\in\mathbb{R}^{d_{model} \times d_k}$, $W^V\in\mathbb{R}^{d_{model}\times d_v}$，$Q, K\in\mathbb{R}^{n\times d_k},\,V\in\mathbb{R}^{n\times d_v}$. 

**Self-attention的计算过程**
- 计算相关性（attention-score）：用当前token的Query与序列中的所有token的Key做点积，计算它们之间的相似度，在序列中体现为矩阵乘法 $QK^\top\in\mathbb{R}^{n\times n}$
- 归一化权重（Softmax）：对得分进行缩放（除以$\sqrt{d_k}$ 保持方差=1，防止点积进入softmax函数的saturated区，进而导致梯度消失），并进行Softmax处理，得到一个总和为1的概率分布。注意这里的softmax是row-wise的，每个token对句中其他token的注意力被归一化为一个和为 $1$，长度为 $n$ 的概率向量，因此 $A\in\mathbb{R}^{n\times n}$.
- 加权求和（Weighted Sum）：根据计算出的权重重新分配当前序列不同token包含的信息，数学上体现为注意力权重矩阵 $A$ 和Value矩阵 $V$ 的矩阵乘法，$\text{Attention}\in\mathbb{R}^{n\times d_{v}}$. 
$$\text{Attention}(Q,K,V) =\text{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V $$
### Multi-head Self-attention
![[Transformer_figure_02.png|200]]
多头注意力把输入序列映射为 $h$ 组不同的Query, Key, Value，分别进行注意力分配计算后堆叠起来，再经过线性变换统一尺度，作为整体的注意力矩阵输出此模块。这样做的好处是，使用不同的方式初始化权重矩阵 $W_Q,W_K,W_V$，能够学习到序列中不同角度的语义信息。

多头注意力在计算上相当于把高维度的自注意力拆分成了多个低维度的并行计算，总计算量其实与相同维度的单头注意力相近，但表达能力更强。需要注意的是多头注意力的参数通常设置为 $d_{model} = h\times d_k$，即embedding向量的维度等于head个数乘权重矩阵的维度。在本文中的设置为 $h=8$, $d_k = d_v = d_{model}/h = 64$.

**Multi-head Self-attention的计算过程**
- 线性投影拆分：输入向量序列 $x$ 在每个head $i$ 分别进行权重矩阵的运算，得到$Q_i=xW_i^Q$, $K_i=xW_i^K$, $V_i=xW_i^V$. 其中 $W_i^Q, W_i^K\in\mathbb{R}^{ d_{model}\times d_k}$, $W_i^V\in\mathbb{R}^{d_{model}\times d_v}$.
- 并行注意力计算：每个头独自计算自注意力 $\text{head}_i=\text{Attention}(Q_i,K_i,V_i)$，得到注意力矩阵的维度为 $\text{head}_i\in\mathbb{R}^{n\times d_{k}}$
- 拼接与最终投影：将所有头的输出拼接起来，$\text{Concat}(\text{head}_1,...,\text{head}_h)$，与权重矩阵 $W^O$ 做乘积，得到最终的自注意力结果。其中$W^O\in\mathbb{R}^{d_{model}\times d_{model}}$ ，这个步骤的目的是为了统一各个头的注意力尺度。
$$
\begin{align}
\text{MultiHead}(Q,K,V)=\text{Concat}(\text{head}_1,...,\text{head}_h)W^O\\
where \quad\text{head}_i=\text{Attention}(xW_i^Q,xW_i^K,xW_i^V)
\end{align}
$$

# Transformer

![[Transformer_figure_03.png|400]]
### Input Embedding
自然语言经过切分、编码转换成一个序列的`token`，此时具体的数据格式是一个整数 `id`。每个不同的 `token` 具有独一无二的 `token_id`. 每个 `token_id` 会被映射到语义向量空间中，得到 $d_{model}$ 维的 embedding vector. 由此得到输入的结构 $x\in\mathbb{R}^{n\times d_{model}}$.
### Positional Encoding
Self-Attention虽然考虑了所有的输入向量，但没有考虑到向量的位置信息。从数学角度看，序列中每个token和目标token做内积的时候无关位置，“注意力”只取决于token的含义，而不受到位置的影响。
因此可以通过位置编码来解决这个问题，Transformer模型将每个token的positional embedding和 input embedding相加，作为整体传递给注意力模块。

本文使用的是 Sinusoidal postional encoding 方案，关于不同 Positional Encoding 模型的笔记在文档 [[PositionalEncoding方法总结]] 中。
### Encoder
本文模型的 Encoder 结构用于从输入序列获取注意力信息，它由 $N=6$ 个 encoder transformer block 堆叠而成。
在一个 Transformer Block 内，输入数据的**流动路线**为：
$$
\begin{align}
\text{Input} = x_{\ell-1} & \rightarrow \text{Attention}(x_{\ell-1})\\
& \rightarrow \text{LayerNorm}(x_{\ell-1}+\text{Attention}(x_{\ell-1}))=\hat{x}_{\ell}\\
& \rightarrow \text{FeedForward}(\hat{x}_{\ell}) \\
& \rightarrow \text{LayerNorm}(\hat{x}_{\ell} + \text{FFN}(\hat{x}_{\ell}))={x}_{\ell}\\
& \rightarrow {x}_{\ell} = \text{Output}
\end{align}
$$

**维度保持**
代表 token embedding 序列的矩阵 $x$ 在这个过程中的维度始终为 $n\times d_{model}$。这种设计允许模型进行极深层的堆叠，并支持残差连接中 $x$ 与子层输出的直接相加，因此在整个 Encoder 结构中，输入输出矩阵的维度保持不变。

**LayerNorm**
在每个子层之后，LayerNorm是针对每个token的，也就是row-wise的。它使得序列中的每一个 Token Embedding 都处于同一个向量空间中（均值为 0，方差为 1）；并且缓解深层网络中的梯度消失或梯度爆炸问题。
关于 Norm 方法的选择和 Pre-Norm 与 Post-Norm 方法的对比在笔记 [[Normalization方法总结]] 中。

**Feed Forward** 
本文的FFN采用两个简单的全链接层
$$
\max(0,xW_1 + b_1)W_2 + b2
$$
### Decoder
本文的 Decoder 结构用于从“已经生成的”部分回答获取注意力信息，结合 Encoder 结构提供的输入的注意力信息，生成对“下一个”输出token的预测。它由 $N=6$ 个 decoder transformer block 堆叠而成，在每个block内数据的流动路线为：
$$
\begin{align}
\text{Input} = x_{\ell-1} & \rightarrow \text{MaskedAttention}(x_{\ell-1})\\
& \rightarrow \text{LayerNorm}(x_{\ell-1}+\text{MaskedAttention}(x_{\ell-1}))=\tilde{x}_{\ell}\\
& \rightarrow \text{CrossedAttention}(\tilde{x}_{\ell},z_{enc})\\
& \rightarrow \text{LayerNorm}(\tilde{x}_{\ell}+\text{CrossedAttention}(\tilde{x}_{\ell},z_{enc}))=\hat{x}_{\ell}\\
& \rightarrow \text{FeedForward}(\hat{x}_{\ell}) \\
& \rightarrow \text{LayerNorm}(\hat{x}_{\ell} + \text{FFN}(\hat{x}_{\ell}))={x}_{\ell}\\
& \rightarrow {x}_{\ell} = \text{Output}
\end{align}
$$

与 encoder transformer block有比较大区别的是$\text{MaskedAttention}$ 和 $\text{CrossedAttention}$ 结构。

**Masked Attention**
这是 Decoder 的第一个自注意力层，它与前面所讲的“一般的” Attention 结构区别在于 Masking。在训练阶段，Decoder是一次性输入整个目标序列的，但是为了模拟推理过程并保持自回归属性，模型在预测target token的时候只能看见这个token之前的单词，而不能偷看未来的单词。
具体方法是，在计算softmax之前，添加上mask矩阵 $M$，它是一个下三角矩阵，所有 $j>i$ 的位置 $M_{ij}=-\infty$，其余位置 $=0$. 被 masked 的自注意力矩阵经过 softmax函数，target token对它位置之后的token的注意力为0.
![[Transformer_figure04.png|400]]
$$
\text{MaskedAttention}(Q,K,V)=\text{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}} + M\right)V 
$$
在训练过程中，一个目标序列可以同时输入，模型会同时计算序列中所有token的预测概率，整个句子的Loss是一次性计算并反向传播的。

**Cross Attention**
交叉注意力结构将输入序列的信息和已有输出序列的信息结合起来，共同作为预测下一个token的依据。具体计算过程中，Query矩阵来自block上一个Masked注意力矩阵的输出$\tilde{x}_{\ell}$，Key和Value矩阵均来自encoder结构的输出$z_{enc}$.
$$
\begin{align}
& Q = \tilde{x}_{\ell}W^Q\\
& K = z_{enc}W^K\\
& V = z_{enc}W^V
\end{align}
$$
需要注意的是，$z_{enc}$ 是输入 $x$ 在 Encoder 中经过$N$ 个block完整运算之后得到的输出。
