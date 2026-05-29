---
title: Attention
description: Self-Attention, Multi-Head Attention, KV Cache, Multi-Query Attention, Group-Query Attention
pubDate: 2026-05-25
topic: Algorithm Notes
tags:
  - transformer
  - attention
---
![[transformer model architechture.png|350]]


# Self-Attention
![[scaled dot-product attention.png|250]]
Self-Attention 是一种处理 Seq2Seq 任务的方法，它的优势是序列中的目标 token 能够获取整个序列的信息，同时还解决了长序列的信息衰减问题。
输入向量的序列 $x\in\mathbb R^{n\times d_{model}}$ 分别经过 3 个权重矩阵，映射为 Query, Key, Value 矩阵，其中$d_{model}$ 表示 token 嵌入向量空间的维度，$n$ 表示序列长度
+ $Q= x W^Q$ 代表当前token“想要寻找什么”
+ $K= x W^K$ 代表序列中每个token分别“提供什么信息”
+ $V=x W^V$ 代表当前token“包含的具体内容”
其中 $W^Q,W^K\in\mathbb{R}^{d_{model}\times d_k},\,Q,K\in\mathbb{R}^{n\times d_k},\, V\in\mathbb{R}^{n\times d_v}$

Self-Attention 的计算过程：
- 计算 token 两两之间的相关性 attention-score：用当前 token 的 Query 与序列中所有 token 的 Key 做点积，计算它们之间的相似度，在序列中体现为矩阵乘法 $QK^\top\in\mathbb{R}^{n\times n}$
- 归一化权重：首先对 attention-score 进行缩放，除以 $\sqrt{d_k}$，其目的是保证矩阵元素的方差等于 $1$，避免进入 sfotmax 函数的 saturated 区继而导致梯度消失。接着进行 row-wise 的 softmax计算，每个 token 对序列中其他 token 的注意力（每行）被转换为一个和为 $1$，长度为 $n$ 的概率向量，因此 $A\in\mathbb{R}^{n\times n}$
- 加权求和：根据计算出的权重重新分配当前序列不同 token 包含的信息，数学上体现为注意力权重矩阵 $A$ 和 Value 矩阵 $V$ 的矩阵乘法
$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V
$$

# Multi-head Attention
![[multi-head attention.png|230]]
多头注意力把输入序列映射为 $h$ 组不同的 Query, Key, Value，分别进行注意力分配计算后堆叠起来，再经过线性变换统一尺度，作为整体的注意力矩阵输出此模块。这样做的好处是，使用不同的方式初始化权重矩阵 $W_Q, W_K, W_V$，能够学习到序列中不同角度的语义信息。
多头注意力再计算上相当于高维度的自注意力拆分成了多个低维度的并行计算，总计算量其实力与相同维度的单头注意力相近，但表达能力更强。需要注意的是多头注意力的参数通常设置为 $d_{model} = h\times d_k$，即 embedding 向量的维度等于 head 数量乘以权重矩阵的维度。再本文中设置为 $h=8,\, d_k=d_v=d_{model}/h = 64$.

Multi-head attention 的计算过程
- 线性投影拆分：输入向量序列 $x$ 在每个 head $i$ 分别进行权重矩阵的运算，得到 $Q_i=x W_i^Q,\,K_i=xW_i^K,\, V_i=xW_i^V$. 其中 $W_i^Q,\, W_i^K\in\mathbb{R}^{d_{model}\times d_{k}}$，$W_i^V\in\mathbb{R}^{d_{model}\times d_v}$.
- 并行注意力计算：每个头独自计算自注意力 $\text{head}_i = \text{Attention}(Q_i, K_i, V_i)$，得到注意力矩阵的维度 $\text{head}_i\in\mathbb{R}^{n\times d_k}$.
- 拼接与最终投影：将所有头的输出拼接起来，$\text{Concat}(\text{head}_1,...,\text{head}_h)$，与权重矩阵 $W^O$ 做乘积，得到最终的自注意力结果。其中 $W^O\in\mathbb{R}^{d_{model}\times d_{model}}$，这个步骤是为了同一各个头的注意力尺度。
$$
\begin{align}
\text{MultiHead}(Q,K,V)=\text{Concat}(\text{head}_1,...,\text{head}_h)W^O\\
where \quad\text{head}_i=\text{Attention}(xW_i^Q,xW_i^K,xW_i^V)
\end{align}
$$
```python
# Multi-Head Attention
import torch
from torch import nn
  
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        # W_q, W_k, W_v, W_o: (d_model, d_model)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, causal_mask=None, padding_mask=None):
        # x: (batch_size, seq_len, d_model)
        # query, key, value: (batch_size, seq_len, d_model)
        query = self.W_q(x)
        key = self.W_k(x)
        value = self.W_v(x)
        
        # query, key, value: (batch_size, num_heads, seq_len, head_dim)
        batch_size = x.size(0)
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # attn_scores: (batch_size, num_heads, seq_len, seq_len)
        attn_scores = torch.matmul(query, key.transpose(-1, -2)) / (self.head_dim ** 0.5)
        if causal_mask is not None:
            attn_scores += causal_mask * -1e9
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)
            attn_scores += padding_mask * -1e9
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # attn_probs @ value: (batch_size, num_heads, seq_len, head_dim)
        output = torch.matmul(attn_probs, value).transpose(-1, -2).contiguous()
        output = output.view(batch_size, self.head_dim * self.num_heads, -1).transpose(-1, -2)
        output = self.W_o(output)
        
        return output
```
# KV Cache
在推理阶段，Decoder 结构需要根据已知 output 序列计算出下一个 token，在这个过程中 Key 矩阵和 Value 矩阵都是可以复用的。具体来讲，假设我们已经得到了 $m$ 个 token 作为 output，我们需要预测第 $m+1$ 个 token，那么数据流向如下
- 输入第 m 个 token 的embedding vector $x_m$ `(1,d_model)`
- 与 $W_q$ 相乘得到 query 向量 $q_m$: `(1,d_model) * (d_model,d_k) -> (1, d_k)`
- 与 $W_k,W_v$ 分别相乘得到 key 向量 $K_m$ 和 value 向量 $V_m$: `(1,d_model) * (d_model,d_k) -> (1,d_k)`, `(1,d_model) * (d_model,d_v) -> (1,d_v)`
- 复用前 $m-1$ 个 token embedding vector $W_k,W_v$ 分别相乘得到 key 矩阵 $K_{1:m-1}$ 和 value 矩阵 $V_{1:m-1}$，经过拼接得到完整的矩阵 $K,V$: `(m-1,d_k) & (1,d_k) -> (m,d_k)`, `(m-1,d_v) & (1,d_v) -> (m,d_v)`
- 计算 attention: `(1,d_k) * (d_k,m) -> (1,m)`, `(1,m) * (m,d_v) -> (1,d_v)`
- multi-head concate: `(1,d_v * h) -> (1,d_model)`
- 经过FNN和归一化给出概率预测 `(1,d_model) * (d_model,vocab_size) -> (1,vocab_size)`
可以注意到，在推理过程中我们复用了K和V，因此用显存换来了计算时间。在使用完整 KVCache的情况下，推理出单个 token的时间复杂度为 $O(n)$，其中 $n$ 表示序列长度，因为单个token的query与整个序列的K和V进行了点积。（在不使用KVCache的情况下，时间复杂度为 $O(n^2)$，因为需要使用整个序列计算 attention）。在 decoder-only 结构中，推理出完整序列的时间复杂度为 $O(p^2+pT)$，其中 $p$ 表示输入序列的长度，$T$ 表示输出序列的长度。

```python
# Multi-Head Attention with KV Cache
import torch
from torch import nn
  
class MHAWithCache(nn.Module):
    def __init__(self, num_heads, d_model):
        super().__init__()
  
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"        
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x, past_key_value=None, use_cache=False):
        batch_size = x.size(0)
        # x: (batch_size, 1, d_model)
        query = self.W_q(x)
        key = self.W_k(x)
        value = self.W_v(x)
        
        # view -> (batch_size, num_heads, 1, head_dim)
        query = query.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # past_key_value: (batch_size, num_heads, seq_len-1, head_dim)
        # concat -> (batch_size, num_heads, seq_len, head_dim)
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=2)
            value = torch.cat([past_value, value], dim=2)
        if use_cache:
            new_past_key_value = (key, value)
            
        # attention_scores: (batch_size, num_heads, 1, seq_len)
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attention_probs = torch.softmax(attention_scores, dim=-1)
        # y: (batch_size, num_heads, 1, head_dim) -> (batch_size, 1, d_model)
        y = torch.matmul(attention_probs, value).transpose(1,2).contiguous()
        y = y.view(batch_size, 1, self.num_heads * self.head_dim)
        y = self.W_o(y)
        
        return (y, new_past_key_value) if use_cache else y
```
# Masked Attention (Causal self-attention)

![[masked attention.png]]
在训练阶段，Decoder 结构一次性输入整个目标序列，但是为了模拟推理过程并保持自回归属性，模型在预测 target token 的时候只能看见这个 token之前的 token，而不能偷看未来的 token。
具体方法是，在计算 softmax 之前，添加上 mask 矩阵 $M$，它是一个下三角矩阵，所有 $j>i$ 的位置 $M_{ij}=-\infty$，其余位置 $=0$. 被 masked 的自注意力矩阵经过 sfotmax 函数，target token 对它位置之后的 token 的注意力为 $0$.
$$
\text{MaskedAttention}(Q,K,V)=\text{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}} + M\right)V 
$$
在训练过程中，一个目标序列可以同时输入，模型会同时计算序列中所有token的预测概率，整个句子的Loss是一次性计算并反向传播的。

# Multi-Query Attention

```python
# Multi-Query Attention
import torch
from torch import nn
  
class MultiQueryAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
  
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, self.head_dim)
        self.W_v = nn.Linear(d_model, self.head_dim)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, causal_mask=None, padding_mask=None):
        # x: (batch_size, seq_len, d_model)
        batch_size = x.size(0)

        # query: (batch_size, seq_len, d_model)
        # key, value: (batch_size, seq_len, head_dim)
        query = self.W_q(x)
        key = self.W_k(x)
        value = self.W_v(x)

        # cutting into multi-heads
        # query: (batch_size, num_heads, seq_len, head_dim)
        # key, value: (batch_size, 1, seq_len, head_dim)
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, 1, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, 1, self.head_dim).transpose(1, 2)

        # attention_score broadcasting: (batch_size, num_heads, seq_len, seq_len)
        attn_scores = torch.matmul(query, key.transpose(-1, -2)) / (self.head_dim ** 0.5)
        if causal_mask is not None:
            attn_scores += causal_mask * -1e9
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)
            attn_scores += padding_mask * -1e9
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # y: (batch_size, seq_len, d_model)
        y = torch.matmul(attn_probs, value).transpose(1, 2).contiguous()
        y = y.view(batch_size, -1, self.num_heads * self.head_dim)
        y = self.W_o(y)

        return y
```

# Group Query Attention

```python
# Group Query Attention
import torch
from torch import nn
  
class GroupQueryAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, group_num):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.group_num = group_num
  
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, group_num * self.head_dim)
        self.W_v = nn.Linear(d_model, group_num * self.head_dim)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, causal_mask=None, padding_mask=None):
        # x: (batch_size, seq_len, d_model)
        batch_size = x.size(0)
        query = self.W_q(x)
        key = self.W_k(x)
        value = self.W_v(x)

        # query: (batch_size, num_heads, seq_len, head_dim)
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)

        # key, value: (batch_size, group_num, seq_len, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        key = key.view(batch_size, -1, self.group_num, self.head_dim).transpose(1,2)
        value = value.view(batch_size, -1, self.group_num,self.head_dim).transpose(1,2)
        repeat_factor = self.num_heads // self.group_num
        key = key.repeat_interleave(repeat_factor, dim=1)
        value = value.repeat_interleave(repeat_factor, dim=1)

        # attn_score: (batch_size, num_heads, seq_len, seq_len)
        attn_score = torch.matmul(query, key.transpose(-1, -2)) / (self.head_dim ** 0.5)
        if causal_mask is not None:
            attn_score += causal_mask * -1e9
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)
            attn_score += padding_mask * -1e9
        attn_prob = torch.softmax(attn_score, dim=-1)
        
        y = torch.matmul(attn_prob, value).transpose(1,2).contiguous()
        y = y.view(batch_size, -1, self.num_heads * self.head_dim)
        y = self.W_o(y)
        
        return y
```