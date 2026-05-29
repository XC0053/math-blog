---
title: Positional Embedding
description: Sinusoidal, RoPE, ALiBi
pubDate: 2026-05-26
topic: Algorithm Notes
tags:
  - transformer
---
# Postional Embedding
Self-attention 结构中不包含任何 token 的位置信息。假设输入 token 表示为 $X=[x_1,x_2,...,x_n],\, x_i\in\mathbb{R}^d$。一层 self-attention 里有 $q_1=x_i W_Q,\, k_i=x_i W_K, \, v_i=x_i W_V$，那么 attention score 为 $s_{ij} = \frac{q_i^{\top}k_j}{\sqrt{d_k}}$。如果没有任何位置信息，那么第 $i$ 个 token 和第 $j$ 个 token 的交互只取决于它们的内容向量，不取决于它们在序列中的位置。
使用 Positional embedding 添加位置信息。Sinusoidal positiondal embedding 将缩放统一过后的绝对位置信息直接添加到输入 embedding 上。RoPE 把位置信息注入到 query/key 的旋转角度里，使 attention score 天然依赖相对距离。ALiBi 不修改 embedding，也不修改 Q/K，而是直接在 attention score 上加一个和距离有关的惩罚项。
# Sinusoidal Positional Embedding
对第 $i$ 个位置的 token，构造一个位置向量 $p_i\in\mathbb{R}^d$，把它直接加到 token embedding 上得到 $z_i=x_i+p_i$，之后模型使用的都是混合后的 $z_i$。
位置向量 $p_i$ 的第 $2m$ 和第 $2m+1$ 维定义为
$$
\begin{aligned}
p_{i, 2m} = \sin\left(\frac{i}{10000^{2m/d}}\right)\\
p_{i, 2m+1} = \cos\left(\frac{i}{10000^{2m/d}}\right)
\end{aligned}
$$
其中 $m = 0,1,...,\frac{d}{2}-1$。也可以写成频率形式，令 $\omega_m=10000^{-2m/d}$，那么
$$
p_{i,2m}=\sin(\omega_m i),\,\, p_{i, 2m+1}=\cos(\omega_m i)
$$
也就是每两个维度构成一组正弦和余弦 $[\sin(\omega_m i),\cos(\omega_m i)]$，不同的 $m$ 对应不同频率，低维度对应高频率，$\omega_m$ 越大，周期 $T_m = \frac{2\pi}{\omega_m}$ 越小，变化越快。

直觉理解：我们可以想象针对 token $x_i\in\mathbb{R}^d$，把它展开写成 $(x_{i1},x_{i2},...,x_{id})$，对应给它的位置编码是 $(p_{i1},p_{i2},...,p_{id})$。可以想象这是一排旋钮，一共 $d$ 个，每个旋钮都有固定的旋转角度，从 $1$ 到 $d$，这个固定的角度逐渐变小。对于 $p_i$，我们将这一排每个旋钮都从0度相位旋转 $i$ 次，也就是 $\omega i$ 的角度。因此这满足 $i+t$ 只是在 $i$ 的基础上额外旋转 $t$ 次。

这样做的优点是：
- 假设我们把 PE 设置为 $1, 2, 3,...$，即每个token的绝对位置的序号，将要面临的问题是序列没有长度限制，PE 的编码没有上限。使用当前方法可以保证任何位置 token 的 PE 处在$[-1,1]^d$ 区间内。
- 假设我们把 PE 设置为$1, 1/2, 1/3,...$，即每个token的绝对位置的序号的倒数，这样可以保证 PE 存在上下界。但是这个方法不满足： $p_{i+t}= f(p_i,t)$。使用 sinusoidal PE可以保证相对位移可以通过线性变换表示。对于某个频率 $\omega$，有三角恒等式 :
$$
\begin{bmatrix} \sin(\omega(i+t))\\ \cos(\omega(i+t)) \end{bmatrix} = \begin{bmatrix} \cos(\omega t) & \sin(\omega t)\\ -\sin(\omega t) & \cos(\omega t) \end{bmatrix} \begin{bmatrix} \sin(\omega i)\\ \cos(\omega i) \end{bmatrix}
$$
	因此从位置 $i$ 的编码，可以通过一个只依赖相对距离 $t$ 的线性变换，得到位置 $i+t$ 的编码。
它的缺点是，位置向量和 token embedding 直接相加，内容信息和位置信息在输入层就被混合在一起，模型需要自己学习如何使用位置结构。此外，它本质上仍然是绝对位置编码；虽然相对位移可以通过三角函数结构线性表示，但 attention score 并没有显式地只依赖相对距离，因此长上下文外推并不一定稳定。

# Rotary Positional Embedding (RoPE)
我们希望设计一种位置编码，不再是简单的把位置向量直接加到 token embedding 上，而是直接对 query vector 和 key vector 进行操作。假设操作分别为 $\tilde{q}_i=f(q_i,i),\, \tilde{k}_j=f(k_j,j)$，经过该操作后 $\tilde{q}_i$ 和 $\tilde{k}_j$ 带有了位置 $i,j$ 的绝对位置信息，并且满足恒等关系 $\langle f(q_i, i),f(k_j,j)\rangle = g(q_i,k_j,i-j)$。
假设某个 token 的 query 和 key 向量分别为 $q_i\in\mathbb{R}^{d_k},\,k_j\in\mathbb{R}^{d_k}$。
RoPE 把向量的每两个维度看成一个二维平面，然后在每个二维平面上旋转一个角度，旋转角度取决于位置 $i$。对于第 $m$ 个二维子空间，频率为 $\theta = 10000^{-2m/d}$，其中 $m=0,1,2,...,\frac{d_k}{2}-1$。不同二维平面的旋转角速度不同，比如第 $m$ 个二维平面的位置 $i$ 上，旋转角度为 $i\theta_m$，那么二维旋转矩阵的为：
$$
R(i\theta_m) = \begin{bmatrix} \cos(i\theta_m) & -\sin(i\theta_m)\\
\sin(i\theta_m) & \cos(i\theta_m) \end{bmatrix} 
$$
$$
\tilde{q}_i^{(m)} = R(i\theta_m) q_i^{(m)} = \begin{bmatrix} \cos(i\theta_m) & -\sin(i\theta_m)\\
\sin(i\theta_m) & \cos(i\theta_m) \end{bmatrix} \begin{bmatrix}q_{i,2m}\\ q_{i, 2m+1}\end{bmatrix}
$$
针对完整 query, key 向量的旋转矩阵 $R_i = \text{diag}(R(i\theta_0),R(i\theta_1),...,R(i\theta_{\frac{d_k}{2}-1}))$，展开即为：
$$
R_i = \begin{bmatrix}
\cos i\theta_0 & -\sin i\theta_0 & 0 & 0 & \cdots & 0 & 0\\
\sin i\theta_0 & cos i\theta_0 & 0 & 0 & \cdots & 0 & 0\\
0 & 0 & \cos i\theta_1 & -\sin i\theta_1 & \cdots & 0 & 0 \\
0 & 0 & \sin i\theta_1 & \cos i\theta_1 & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & \cos i\theta_{d_k/2-1} & -\sin i\theta_{d_k/2-1}\\
0 & 0 & 0 & 0 & \cdots & \sin i\theta_{d_k/2-1} & \cos i\theta_{d_k/2-1}
\end{bmatrix}
$$
那么 attention score 为：
$$
s_{ij} = \frac{\tilde{q}_i^{\top}\tilde{k}_j}{\sqrt{d_k}} = \frac{(R_iq_i)^{\top}(R_j k_j)}{\sqrt{d_k}}
 = \frac{q_i^{\top} R_i^{\top }R_jk_j}{\sqrt{d_k}} = \frac{q_i^{\top}R_{j-i}k_j}{\sqrt{d_k}}$$
 因为二维旋转矩阵满足 $R(\alpha)^{\top}R(\beta) = R(\beta-\alpha)$，所以旋转矩阵满足 $R_i^{\top }R_j = R_{j-i}$
 
 RoPE 的优点是：
 - 它不会把位置向量直接加到 token embedding 上，因此内容和位置的混合更干净
 - 它让 attention score 显式依赖相对距离：$s_{ij}\propto q_i^{\top}R_{j-i}k_j$.
 - 它保留了绝对位置信息。因为 $q_i​$ 和 $k_j$​ 在进入点积前分别经过了 $R_i​$ 和 $R_j​$，所以每个位置仍然有自己的相位。
# ALiBi Positional Emebdding
ALiBi 不构造位置 embedding，也不旋转 Q/K，而是直接修改 attention score。普通的 causal attention 中，第 $i$ 个位置关注第 $j$ 个位置的 score 是：
$$s_{ij} = \frac{{q}_i^{\top}{k}_j}{\sqrt{d_k}}$$
ALiBi 将其改为：
$$
s_{ij} = \frac{{q}_i^{\top}{k}_j}{\sqrt{d_k}} - m_h(i-j)
$$
其中 $i\geq j$ ，因为当前位置 $i$ 只能看过去位置 $j$，距离越大惩罚越大，也就是说相距越远的 token 越不被关注；$m_h>0$ 是第 $h$ 个 attention head 的 slope，不同 head 使用不同的斜率（以提高表达能力，不同 attention head 具有不同的感受野）。
ALiBi 对 softmax 的影响：
$$
\alpha_{ij}^{(h)} = \frac{\exp (s_{ij}^{(h)})}{\sum_{t\leq i}\exp(s_{it}^{(h)})} = \frac{ \exp\left(\frac{q_i^\top k_j}{\sqrt{d_k}} - m_h(i-j)\right) }{ \sum_{t\le i} \exp\left(\frac{q_i^\top k_t}{\sqrt{d_k}} - m_h(i-t)\right) }
$$
可以拆成：
$$\exp\left(\frac{q_i^\top k_j}{\sqrt{d_k}}\right) \cdot \exp(-m_h(i-j))$$
所以 ALiBi 相当于给距离为 $i-j$的位置乘了一个指数衰减因子：
$$\exp(-m_h(i-j))$$
这说明它虽然在 score 上是线性 bias，但经过 softmax 后，对注意力权重的影响是指数衰减。
ALiBi 的特点是：
- 简单，不需要 positional embedding，不需要旋转矩阵；但是位置信息表达能力单一
- 支持任意长度的序列，外推性很好

