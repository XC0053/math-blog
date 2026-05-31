---
title: Proximal Policy Optimization (PPO)
description:
pubDate: 2026-05-31
topic: Algorithm Notes
tags:
  - reinforcement-learning
  - rlhf
---
PPO 可以理解为一种更稳定的 Policy Gradient 方法。它的核心问题是：我们希望提高好动作的概率、降低坏动作的概率，但每次更新又不能让新策略偏离旧策略太远。
因此 PPO 的逻辑链可以分成三步：
1. Policy Gradient 需要一个信号判断动作好坏。
2. Actor-Critic 用 advantage 作为动作好坏的估计。
3. PPO 在使用 advantage 更新策略时，通过 clipping 限制策略变化幅度。

### PPO 数学公式推理
##### 从 Policy Gradient 到 Advantage

上接 [[Policy Gradient]] 笔记，Policy Gradient 的基本形式可以写成：
$$
\nabla J(\pi_\theta)  =\mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=0}^{T-1}\Psi_t \nabla\log \pi_{\theta}(a_t|s_t)\right]
$$
其中 $\nabla_{\theta}\log \pi_{\theta}(a_t\mid s_t)$ 表示提高当前策略在状态 $s_t$ 下选择动作 $a_t$ 的概率的方向。真正关键的是 $\Psi_t$：它决定当前动作应该被增强还是削弱。如果 $\Psi_t>0$，说明这个动作应该更容易被选中；如果 $\Psi_t<0$，说明这个动作应该被压低概率。

最原始的 REINFORCE 方法直接令 $\Psi_t=R(\tau)$，也就是用整条轨迹的总回报评价每一步动作。但这太粗糙，因为一条轨迹最终奖励高，并不代表其中每一步动作都好。
****
##### 从整条轨迹 reward 到 reward-to-go

为了让第 t 步动作只对它之后的结果负责，可以使用从当前时刻开始的折扣回报 $G_t=\sum_{l=0}^{T-t-1}\gamma^l r_{t+l}$，也可以递归写成
$$
G_t=r_t+\gamma G_{t+1}.
$$
相比整条轨迹奖励 $R(\tau)$，reward-to-go $G_t$ 更合理，因为当前动作只对当前及未来奖励负责，而不应该对过去已经发生的奖励负责。

但是 $G_t$ 仍然是单条采样轨迹上的实际回报，随机性很强。如果直接用它作为动作好坏的估计，梯度方差会比较大。
****
##### 从 reward-to-go 到 advantage

为了更精确地判断一个动作是否值得增强，引入动作价值函数和状态价值函数。

- 动作价值函数 $Q^\pi(s_t,a_t)=\mathbb{E}_{\pi}[G_t\mid s_t,a_t]$ 表示：在状态 $s_t$ 下选择动作 $a_t$，之后继续按照策略 $\pi$ 执行，未来期望能得到多少回报。
- 状态价值函数 $V^\pi(s_t)=\mathbb{E}_{a_t\sim\pi(\cdot\mid s_t)}[Q^\pi(s_t,a_t)]=\mathbb{E}_{\pi}[G_t\mid s_t]$ 表示：在状态 $s_t$ 下按照当前策略平均行动，未来期望能得到多少回报。
- advantage $A^\pi(s_t,a_t)=Q^\pi(s_t,a_t)-V^\pi(s_t)$ 表示：当前动作的期望回报，比当前状态下的平均动作水平高多少。

如果 $A^\pi(s_t,a_t)>0$，说明这个动作比当前状态下的平均水平更好，应该提高它的概率。如果 $A^\pi(s_t,a_t)<0$，说明这个动作比平均水平更差，应该降低它的概率。
因此 Policy Gradient 可以写成：
$$
\nabla_{\theta}J(\theta)
=
\mathbb{E}_{\tau\sim\pi_\theta}
\left[
\sum_{t=0}^{T-1}
A^\pi(s_t,a_t)
\nabla_{\theta}\log \pi_{\theta}(a_t\mid s_t)
\right].
$$
****
##### PPO 中如何估计 advantage

真实的 $Q^\pi(s_t,a_t)$ 和 $V^\pi(s_t)$ 都无法直接知道，因为它们都是对未来所有可能轨迹的期望。在 PPO / Actor-Critic 中，通常训练一个 critic 来估计状态价值函数，即 $V_\phi(s_t)\approx V^\pi(s_t)$。

然后用 TD error 构造 advantage 的估计。单步 TD error 定义为 $\delta_t=r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t)$。它可以理解为：当前真实拿到的奖励 $r_t$，加上下一个状态的估计价值，是否比当前状态的估计价值更高。

如果 $\delta_t>0$，说明这一步动作带来的结果比 critic 原本预期更好；如果 $\delta_t<0$，说明这一步动作带来的结果比 critic 原本预期更差。因此，单步 TD error 可以作为 advantage 的一种估计，即 $\hat{A}^{(1)}_t=\delta_t$。

但是单步 TD error 只使用一步真实奖励，后续长期价值完全依赖 critic 估计。因此它的优点是方差较小，缺点是如果 critic 不准，偏差会比较大。

另一种极端做法是使用完整的 reward-to-go，即 $\hat{A}^{\mathrm{full}}_t=G_t-V_\phi(s_t)$。这种方法更少依赖 critic 的 bootstrap 估计，但是 $G_t$ 来自一整条采样轨迹，轨迹本身随机性很强，因此方差较大。
****
##### GAE：在单步 TD 和完整回报之间折中

为了平衡单步 TD error 和完整 reward-to-go，可以使用 n-step advantage estimator。它使用前 $n$ 步真实奖励，然后用 critic 估计第 $n$ 步之后的长期价值：
$$
\hat{A}^{(n)}_t
=
\sum_{l=0}^{n-1}\gamma^l r_{t+l}
+
\gamma^n V_\phi(s_{t+n})
-
V_\phi(s_t).
$$
等价地，它也可以写成多个 TD error 的折扣和：$\hat{A}^{(n)}_t=\sum_{l=0}^{n-1}\gamma^l \delta_{t+l}$。

不同的 $n$ 对应不同的 bias-variance trade-off：$n$ 越小，越依赖 critic，方差较小，但偏差可能较大；$n$ 越大，越依赖真实采样回报，偏差较小，但方差可能较大。

GAE 不手动选择某一个固定的 $n$，而是把当前及未来多个 TD error 按照衰减权重累加：
$$
\hat{A}^{\mathrm{GAE}(\gamma,\lambda)}_t
=
\sum_{l=0}^{T-t-1}
(\gamma\lambda)^l
\delta_{t+l}.
$$
其中 $\lambda$ 控制折中程度。当 $\lambda\to 0$ 时，GAE 更接近单步 TD error，方差较小，但更依赖 critic；当 $\lambda\to 1$ 时，GAE 更接近完整 reward-to-go，方差较大，但更少依赖 critic。

因此 GAE 的作用是：给 PPO 提供一个更稳定的 advantage 估计。

****
##### PPO 为什么要限制策略更新幅度

有了 advantage 之后，策略更新的直觉是：如果 $\hat{A}_t>0$，提高 $\pi_\theta(a_t\mid s_t)$；如果 $\hat{A}_t<0$，降低 $\pi_\theta(a_t\mid s_t)$。

但是如果策略一次更新太大，新策略可能会偏离旧策略太远，导致之前采样得到的数据不再可靠，训练容易不稳定。

PPO 用新旧策略概率比值衡量策略变化：

$$
r_t(\theta)
=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}.
$$

如果 $r_t(\theta)>1$，说明新策略比旧策略更倾向于选择这个动作。如果 $r_t(\theta)<1$，说明新策略比旧策略更不倾向于选择这个动作。

普通的 policy gradient surrogate objective 可以写成 $L^{\mathrm{PG}}(\theta)=\mathbb{E}_t[r_t(\theta)\hat{A}_t]$，但这个目标没有限制 $r_t(\theta)$ 的变化幅度，所以 PPO 引入 clipping：

$$
L^{\mathrm{CLIP}}(\theta)
=
\mathbb{E}_t
\left[
\min
\left(
r_t(\theta)\hat{A}_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t
\right)
\right].
$$

它的作用是限制新策略相对旧策略的变化，即 $r_t(\theta)\in[1-\epsilon,1+\epsilon]$。

如果 advantage 为正，PPO 不允许策略无限制地提高该动作概率。如果 advantage 为负，PPO 不允许策略无限制地降低该动作概率。因此 PPO 的核心不是单纯“提高好动作概率、降低坏动作概率”，而是在这个过程中限制每次策略更新的幅度，使训练更加稳定。

****
##### PPO 的整体训练逻辑

PPO 的整体流程可以概括为：

1. 使用旧策略 $\pi_{\theta_{\mathrm{old}}}$ 采样轨迹。
2. 使用 critic $V_\phi(s_t)$ 计算 TD error：$\delta_t=r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t)$。
3. 使用 GAE 计算 advantage：$\hat{A}^{\mathrm{GAE}}_t=\sum_{l=0}^{T-t-1}(\gamma\lambda)^l\delta_{t+l}$。
4. 使用 clipped objective 更新 actor。
5. 使用 value loss 更新 critic，例如 $L^{\mathrm{VF}}(\phi)=\mathbb{E}_t[(V_\phi(s_t)-\hat{R}_t)^2]$，其中 $\hat{R}_t$ 可以理解为 critic 的训练目标，常见做法是 $\hat{R}_t=\hat{A}_t+V_\phi(s_t)$。

其中 actor 的 clipped objective 为：
$$
L^{\mathrm{CLIP}}(\theta)
=
\mathbb{E}_t
\left[
\min
\left(
r_t(\theta)\hat{A}_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t
\right)
\right].
$$
整体上，PPO 可以概括为：Policy Gradient + Advantage Estimation + Clipped Policy Update。

### 细节解释
#### 从 Advantage 推到 TD Error
这一部分说明为什么 TD error 可以作为 advantage 的单步采样估计。

根据定义，$A^\pi(s_t,a_t)=Q^\pi(s_t,a_t)-V^\pi(s_t)$。又因为 $Q^\pi(s_t,a_t)=\mathbb{E}_{\pi}[G_t\mid s_t,a_t]$，所以 $A^\pi(s_t,a_t)=\mathbb{E}_{\pi}[G_t\mid s_t,a_t]-V^\pi(s_t)$。

由 reward-to-go 的递推式 $G_t=r_t+\gamma G_{t+1}$ 代入可得：
$$
A^\pi(s_t,a_t)
=
\mathbb{E}_{\pi}[r_t+\gamma G_{t+1}\mid s_t,a_t]
-
V^\pi(s_t).
$$
给定 $s_t,a_t$ 后，下一个状态由环境转移概率产生，即 $s_{t+1}\sim P(\cdot\mid s_t,a_t)$。因此可以先对 $s_{t+1}$ 条件化。根据马尔可夫性，已知 $s_{t+1}$ 后，未来轨迹只取决于 $s_{t+1}$ 和后续策略 $\pi$，于是：
$$
\mathbb{E}_{\pi}[G_{t+1}\mid s_t,a_t,s_{t+1}]
=
\mathbb{E}_{\pi}[G_{t+1}\mid s_{t+1}]
=
V^\pi(s_{t+1}).
$$
所以：

$$
A^\pi(s_t,a_t)
=
\mathbb{E}_{s_{t+1}\sim P}
\left[
r_t+\gamma V^\pi(s_{t+1})-V^\pi(s_t)
\mid s_t,a_t
\right].
$$
定义单步 TD error 为 $\delta_t=r_t+\gamma V^\pi(s_{t+1})-V^\pi(s_t)$，于是得到 $A^\pi(s_t,a_t)=\mathbb{E}[\delta_t\mid s_t,a_t]$。这说明 TD error 的条件期望等于真实 advantage，因此单步 TD error 可以看成 advantage 的单步采样估计。

#### $n$-step Advantage 与 GAE 展开
单步 advantage 估计为 $\hat{A}^{(1)}_t=\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$。

两步 TD error 的展开为：
$$
\delta_t+\gamma\delta_{t+1}
=
r_t+\gamma r_{t+1}
+\gamma^2V(s_{t+2})
-V(s_t).
$$
因此两步 advantage 估计为 $\hat{A}^{(2)}_t=r_t+\gamma r_{t+1}+\gamma^2 V(s_{t+2})-V(s_t)$。

推广到 n-step：
$$
\hat{A}^{(n)}_t
=
\sum_{l=0}^{n-1}\gamma^l r_{t+l}
+
\gamma^nV(s_{t+n})
-V(s_t).
$$
它也等价于多个 TD error 的折扣和，即 $\hat{A}^{(n)}_t=\sum_{l=0}^{n-1}\gamma^l\delta_{t+l}$。

GAE 将不同长度的 n-step advantage 进行指数加权，相当于把当前及未来 TD error 按照 $(\gamma\lambda)^l$ 衰减累加：
$$
\hat{A}^{\mathrm{GAE}(\gamma,\lambda)}_t
=
\sum_{l=0}^{T-t-1}(\gamma\lambda)^l\delta_{t+l}.
$$
#### 为什么通常估计 $V^{\pi}(s_t)$ 而不是 $Q^\pi(s_t,a_t)$
 
 Advantage 的定义是 $A^\pi(s_t,a_t)=Q^\pi(s_t,a_t)-V^\pi(s_t)$。理论上可以直接估计 $Q^\pi(s_t,a_t)$，但在 PPO / Actor-Critic 中通常估计 $V^\pi(s_t)$。

原因是 $V(s)$ 只需要输入状态，学习“当前策略平均来说这个状态值多少钱”。而 $Q(s,a)$ 需要同时输入状态和动作，学习“在这个状态下选择某个具体动作值多少钱”。在高维动作空间、连续动作空间，尤其是语言模型的 token 空间中，直接估计 $Q(s,a)$ 更困难，也更难覆盖所有动作。

因此实践中通常训练 critic $V_\phi(s_t)\approx V^\pi(s_t)$，再通过 TD error 和 GAE 构造 advantage 估计。