---
title: Policy Gradient
description: Policy gradient methods are a class of reinforcement learning algorithms and a sub-class of policy optimization methods.
pubDate: 2026-05-29
topic: Algorithm Notes
tags:
  - reinforcement-learning
---
先定义一条轨迹 $\tau = (s_0, a_0, s_1, a_1, ...,s_T)$，其中 $R(\tau)$ 表示这整条轨迹的累积奖励，通常可以是 $R(\tau) = \sum_{t=0}^{T-1}r_t$，或者带折扣的 $R(\tau) = \sum_{t=0}^{T-1}\gamma^t r_t$。
优化目标为 
$$
\arg\max_{\pi_theta} J(\pi_{\theta}) = \arg\max_{\pi_theta}\mathbb{E}_{\tau\sim\pi_{\theta}}[R(\tau)]
$$
也就是我们希望找到一个策略 $\pi_\theta$ 使得它采样出来的轨迹平均奖励最大。因为轨迹是根据采样策略得到的，所以期望可以展开成：
$$
J(\pi_\theta) = \sum_{\tau} R(\tau) P(\tau|\pi_{\theta})
$$
也就是：每一条可能轨迹的奖励 * 这条轨迹在当前策略下出现的概率，求和

接下来要求梯度
$$
\nabla J(\pi_\theta) = \nabla\sum_{\tau}R(\tau)P(\tau|\pi_\theta)
$$
因为 $R(\tau)$ 本身是这条轨迹已经发生后的奖励，不直接依赖参数 $\theta$，所以梯度只作用在轨迹概率上，因此
$$
\nabla J(\pi_\theta) = \sum_{\tau}R(\tau)\nabla P(\tau|\pi_\theta)
$$
这里的关键是：策略参数 $\theta$ 改变以后，不直接改变某条固定轨迹的奖励，而是改变哪些轨迹更容易被采样出来。因此优化策略的本质是，让搞奖励的轨迹出现概率更高，让低奖励轨迹出现概率更低。
接下来，使用 log-derivative trick 进行等式变换可以得到：
$$
\begin{aligned}
\nabla P(\tau|\pi_\theta) & = P(\tau|\pi_\theta)\frac{\nabla  P(\tau|\pi_\theta)}{ P(\tau|\pi_\theta)} =P(\tau|\pi_\theta)\nabla\log P(\tau|\pi_\theta)\\
\nabla J(\pi_\theta) & = \sum_{\tau} R(\tau)P(\tau|\pi_\theta)\nabla\log P(\tau|\pi_\theta)\\
 & =\mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)\nabla\log P(\tau|\pi_\theta)]
\end{aligned}\tag{1}
$$
轨迹是根据策略得到的系列状态和动作，所以可以展开式子：
$$
P(\tau|\pi_\theta) = \rho_0(s_0)\prod_{t=0}^{T-1} P(s_{t+1} | s_t, a_t)\pi_{\theta}(a_t|s_t)
$$
其中，$\rho_0(s_0)$ 表示初始状态 $s_0$ 的先验分布，$\pi_{\theta}(a_t|s_t)$ 表示策略在状态 $s_t$ 下选择动作 $a_t$ 的概率，$P(s_{t+1} | s_t, a_t)$ 表示环境在状态 $s_t$ 执行动作 $a_t$ 后，转移到 $s_{t+1}$ 的概率。所以一整条轨迹的概率可以表示为：
$$
初始状态概率 \times 每一步选中动作的概率 \times 每一步环境转移的概率
$$
然后对它取 log 得到：
$$
\log P(\tau|\pi_{\theta}) = \log \rho_0(s_0) + \sum_{t=0}^{T-1}\log P(s_{t+1}|s_t, a_t) +\sum_{t=0}^{T-1}\log \pi_{\theta}(a_t|s_t)
$$
然后对 $\theta$ 求梯度，由于前 2 项均与 $\theta$ 无关，因此得到：
$$
\begin{aligned}
\nabla \log P(\tau|\pi_{\theta}) & = \nabla\left[
\log \rho_0(s_0) + \sum_{t=0}^{T-1}\log P(s_{t+1}|s_t, a_t) +\sum_{t=0}^{T-1}\log \pi_{\theta}(a_t|s_t)
\right] \\ 
& = \sum_{t=0}^{T-1}\nabla\log \pi_{\theta}(a_t|s_t)
\end{aligned}
$$
将结果代入 $(1)$ 得到 Policy Gradient
$$
\nabla J(\pi_\theta)  =\mathbb{E}_{\tau\sim\pi_\theta}\left[R(\tau)\sum_{t=0}^{T-1}\nabla\log \pi_{\theta}(a_t|s_t)\right]\tag{2}
$$
这个式子的含义是，如果一条轨迹的总奖励 $R(\tau)$ 很高，那么这条轨迹中所有动作的 log probability 都应该被提高。这是一个相对原始的版本，因为它把整条轨迹的总奖励分配给了轨迹中的所有动作，也就是说不管某个动作到底对最终奖励贡献大不大，它都会被同一个 $R(\tau)$ 加权。在实际引用中会使用 $\Psi_t$，它评价的是在第 $t$ 步，动作 $a_t$ 应该被强化还是被削弱，不同的模型用不同方法定义 $\Psi_t$。
$$
\nabla J(\pi_\theta)  =\mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=0}^{T-1}\Psi_t\nabla\log \pi_{\theta}(a_t|s_t)\right]
$$

- **Vanilla REINFORCE**  使用整条轨迹累积奖励 $\Psi_t = \sum_{t=0}^{\infty} r_t = R(\tau)$
- **Reward-to-GO REINFORCE** 使用从当前步开始后轨迹的折扣奖励 $\Psi_t = \sum_{t'=t}^{\infty}\gamma^{t'-t}r_{t'}$
- **REINFORCE with Baseline** 引入 baseline $\Psi_t = \sum_{t'=t}^{\infty}\gamma^{t'-t}r_{t'} - b(s_t)$，这里的 baseline $b(s_t)=V^{\pi}(s_t)$ ，也就是减去这个状态的平均奖励
- **Q Actor-Critic** 用动作价值函数 $\Psi_t=Q^{\pi}(s_t, a_t)$，其中 $Q^{\pi}(s_t, a_t) = \mathbb{E}[\sum_{k=t}^{T-1} \gamma^{k-t}r_k|s_t, a_t]$，它表示在状态 $s_t$ 下选择动作 $a_t$，然后继续按照策略 $\pi$ 执行，最终期望能拿到多少长期汇报
- **Advantage Actor-Critic** 优势函数 $\Psi_t =A^{\pi}(s_t, a_t)$，其中 $A^{\pi}(s_t, a_t)=Q^{\pi}(s_t, a_t)-V^{\pi}(s_t)$，它表示在状态 $s_t$ 下，动作 $a_t$ 相比这个状态的平均动作好多少，其中 $V^{\pi}(s_t)=\mathbb{E}_{a\sim \pi}[Q^{\pi}(s_t,a)]$ 表示状态价值，也就是在状态 $s_t$​ 下按照当前策略继续行动的平均回报
- **TD Actor-Critc** 使用时序差分残差 $\Psi_t = r_t + \gamma V^{\pi}(s_{t+1})-V^{\pi}(s_t)$，他表示当前一步得到的奖励加上下一个状态的估计价值，和当前状态原本估计价值之间的差距，也就是“新的估计 - 旧的估计”。