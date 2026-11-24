---
module: "04B"
title: "Advanced Deep Learning & Reinforcement Learning"
difficulty: "advanced"
estimated_hours: 16
prerequisites:
  - "04A"
learning_levels:
  - foundation
  - applied
  - production
topics:
  - distributed-training
  - ddp-fsdp
  - mixed-precision
  - reinforcement-learning
  - policy-gradients
  - ppo-grpo
---

# Module 04B: Advanced Deep Learning & Reinforcement Learning

[← Previous: Module 04A](./04a_deep_learning_and_pytorch.md) | [Phase 2 Index](./README.md) | [Next: Phase 3 / Module 05A →](../03-modern-ai-and-genai/05a_nlp_and_language_models.md)

---

## 1. Overview

Scaling neural systems beyond single-GPU workstations requires distributed execution paradigms (Distributed Data Parallel and Fully Sharded Data Parallel) and post-training alignment techniques. 

This advanced module covers distributed training topologies, mixed-precision memory optimization (BF16, FP8, ZeRO memory stages), and reinforcement learning fundamentals—spanning Markov Decision Processes (MDPs), REINFORCE policy gradients, Proximal Policy Optimization (PPO), and modern LLM alignment algorithms (RLHF, DPO, GRPO).

---

## 2. Who Should Study This?

- **AI Infrastructure & Systems Engineers** scaling model training and fine-tuning across multi-GPU clusters.
- **LLM Alignment Engineers** preparing to implement Direct Preference Optimization (DPO) and Group Relative Policy Optimization (GRPO).
- **Advanced ML Practitioners** requiring a deep understanding of distributed sharding and reinforcement learning algorithms.

---

## 3. Learning Objectives

By the end of this module, you will be able to:
- **Architect distributed training topologies**, distinguishing between Distributed Data Parallel (DDP), Tensor Parallel (TP), and Fully Sharded Data Parallel (FSDP).
- **Calculate GPU memory overhead**, dissecting model parameters, gradients, optimizer states, and activations across ZeRO stages 1, 2, and 3.
- **Implement the REINFORCE policy gradient algorithm** in PyTorch for discrete action environments.
- **Explain modern LLM alignment algorithms** (PPO, DPO, GRPO), understanding how token reward signals shape generative distributions without reward hacking.

---

## 4. Prerequisites

- **Core Deep Learning & Autograd:** PyTorch training loops and computational graphs from [Module 04A](./04a_deep_learning_and_pytorch.md).
- **Probability & Optimization:** Expected values, policy distributions, and SGD from [Module 02](../01-foundations/02_math_and_statistics.md).

---

## 5. Learning Levels

### Foundation
- Understand Markov Decision Processes (state, action, transition probability, reward, discount factor $\gamma$).
- Distinguish between value-based methods (DQN) and policy-based methods (Policy Gradients).

### Applied Engineering
- Implement REINFORCE policy gradient training with reward-to-go and baseline subtraction.
- Configure Distributed Data Parallel (`torch.nn.parallel.DistributedDataParallel`) using `torchrun`.

### Production Engineering
- Calculate multi-GPU VRAM budgets for billion-parameter model training.
- Prevent gradient variance explosion in PPO using clipped surrogate objectives.

---

## 6. Core Concepts

### 6.1 GPU Memory Math: The 16x Parameter Overhead
When training in standard FP32 with AdamW, memory consumption far exceeds raw model weights:
- **Weights (FP32):** 4 bytes per parameter.
- **Gradients (FP32):** 4 bytes per parameter.
- **AdamW Optimizer States:** 8 bytes per parameter (4 bytes for first moment $m_t$, 4 bytes for second moment $v_t$).
- **Total Static Overhead:** $16 	ext{ bytes per parameter!}$

A 7-billion parameter model requires $7 	imes 16 = 112	ext{ GB}$ of VRAM just to store static optimizer states, before allocating memory for activations or KV cache.

### 6.2 ZeRO Memory Sharding Stages (DeepSpeed / FSDP)
- **ZeRO-1:** Shards optimizer states across data-parallel ranks ($4	imes$ memory reduction).
- **ZeRO-2:** Shards optimizer states and gradients ($8	imes$ memory reduction).
- **ZeRO-3 (FSDP):** Shards optimizer states, gradients, and model parameters. Each GPU fetches weights on-the-fly during forward/backward passes.

---

## 7. Technical Deep Dive

### 7.1 Policy Gradient Theorem & The REINFORCE Objective
In reinforcement learning, the agent aims to maximize expected discounted return:
$$J(	heta) = \mathbb{E}_{	au \sim \pi_	heta} [R(	au)]$$

The **Policy Gradient Theorem** allows us to compute the gradient of expected return without knowing the environment transition dynamics:
$$
abla_	heta J(	heta) = \mathbb{E}_{	au \sim \pi_	heta} \left[ \sum_{t=0}^T 
abla_	heta \log \pi_	heta(a_t \mid s_t) G_t ight]$$

Where $G_t = \sum_{k=t}^T \gamma^{k-t} r_k$ is the **Return-to-Go**.

---

## 8. Practical Implementation

Below is a complete, runnable REINFORCE policy gradient training step implemented in PyTorch:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# 1. Policy Network (Maps observation to action logits)
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim)
        )

    def forward(self, obs: torch.Tensor) -> Categorical:
        logits = self.net(obs)
        return Categorical(logits=logits)

# 2. Simulate One Trajectory Rollout & Update
policy = PolicyNetwork(obs_dim=4, action_dim=2)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)

# Simulated rollout data: observations, actions, rewards
saved_log_probs = []
rewards = [1.0, 1.0, 1.0, 0.0]  # Trajectory rewards

for _ in range(len(rewards)):
    mock_obs = torch.randn(4)
    dist = policy(mock_obs)
    action = dist.sample()
    saved_log_probs.append(dist.log_prob(action))

# 3. Compute Discounted Return-to-Go
gamma = 0.99
returns = []
G = 0.0
for r in reversed(rewards):
    G = r + gamma * G
    returns.insert(0, G)

returns = torch.tensor(returns)
# Normalize returns to stabilize gradient updates
returns = (returns - returns.mean()) / (returns.std() + 1e-8)

# 4. Policy Gradient Loss
policy_loss = []
for log_prob, G_t in zip(saved_log_probs, returns):
    policy_loss.append(-log_prob * G_t)

optimizer.zero_grad()
total_loss = torch.stack(policy_loss).sum()
total_loss.backward()
optimizer.step()

print(f"Policy Gradient Step Successful! Loss: {total_loss.item():.4f}")
```

---

## 9. Hands-on Exercises

### Beginner
Calculate the exact GPU memory required to train a 13-billion parameter model under:
1. Pure FP32 training with AdamW.
2. Mixed-precision BF16 with AdamW.
3. ZeRO-3 parameter, gradient, and optimizer state sharding across 8 GPUs.

### Intermediate
Implement baseline subtraction ($G_t - b(s_t)$) in the REINFORCE script above and verify mathematically that it reduces gradient variance without introducing bias.

### Advanced
Derive the PPO Clipped Surrogate Objective:
$$L^{	ext{CLIP}}(	heta) = \hat{\mathbb{E}}_t \left[ \min\left(r_t(	heta)\hat{A}_t, \, 	ext{clip}(r_t(	heta), 1-\epsilon, 1+\epsilon)\hat{A}_tight) ight]$$
Explain why clipping prevents catastrophically large policy divergence.

---

## 10. Mini Project: Distributed Training Simulation

Build a multi-GPU training simulation:
1. Use `torch.multiprocessing` to spawn 2 simulated worker ranks.
2. Implement gradient averaging across ranks using all-reduce logic.
3. Verify that model weights remain exactly synchronized across ranks after each optimization step.

---

## 11. Production Considerations

- **Communication Bottlenecks:** In DDP and FSDP, inter-GPU interconnect bandwidth (NVLink vs. standard PCIe) is frequently the primary training bottleneck, not raw GPU FLOPs.
- **Gradient Accumulation:** When batch sizes cannot fit in VRAM, simulate large effective batch sizes by calling `loss.backward()` across $N$ mini-batches before calling `optimizer.step()`.
- **Reward Hacking in RLHF:** Language models optimize rewards aggressively, learning to generate verbose, flattering, or sycophantic text that scores high on reward models while degrading human utility.

---

## 12. Common Mistakes & Gotchas

1. **Not Shuffling Distributed Datasets per Epoch:** In DDP, forgetting to call `sampler.set_epoch(epoch)` causes each worker to process the exact same order of batches every epoch.
2. **Ignoring Gradient Clipping in RL:** Policy gradients exhibit extreme variance; training without `clip_grad_norm_` causes catastrophic policy collapse.
3. **Confusing DDP with FSDP:** DDP replicates the entire model on every GPU and only averages gradients; FSDP shards model weights, gradients, and optimizer states across GPUs.

---

## 13. Interview Questions & System Design

1. **Q:** What is Direct Preference Optimization (DPO), and how does it eliminate the separate reward model and reinforcement learning training loop of RLHF?  
   **A:** DPO reparameterizes the reward function directly in terms of the implicit policy $\pi_	heta$ and reference model $\pi_{	ext{ref}}$, optimizing a closed-form cross-entropy loss over chosen and rejected pairs without training a separate reward model or executing unstable PPO loops.
2. **System Design:** Design a distributed fine-tuning cluster to align a 70B parameter model across 16 NVIDIA H100 GPUs using FSDP and FlashAttention-2.

---

## 14. Recommended Resources

- [DeepSpeed ZeRO Paper](https://arxiv.org/abs/1910.02054) — Rajbhandari et al.
- [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) — Schulman et al.
- [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290) — Rafailov et al.

---

## 15. Completion Checklist

- [ ] Can calculate static and dynamic GPU memory overhead for LLM training.
- [ ] Understand differences between DDP, ZeRO-1, ZeRO-2, and FSDP.
- [ ] Understand the policy gradient theorem and REINFORCE algorithm.
- [ ] Understand why modern post-training alignment leverages DPO and GRPO.

---

## 16. Further Exploration

- Study Group Relative Policy Optimization (GRPO) used in DeepSeek-R1 for mathematical and code reasoning alignment without a critic network.
- Explore FlashAttention-2 tiling and SRAM memory IO optimization.
