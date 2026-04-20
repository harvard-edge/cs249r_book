"""
MLPerf EDU: Micro-RL — Reinforcement Learning Workload
=======================================================
Provenance: Williams 1992, "Simple Statistical Gradient-Following
            Algorithms for Connectionist Reinforcement Learning" (REINFORCE)
Maps to: Emerging RL benchmarks in MLPerf / MLCommons

This implements a simple policy gradient agent on locally-simulated
environments — no OpenAI Gym dependency required. The environments
are pure-Python implementations suitable for educational use.

Pedagogical concepts:
- Policy gradient theorem
- Reward discounting and baselines
- Exploration vs exploitation
- On-policy vs off-policy learning

Architecture:
    PolicyNet: Linear(state_dim, 64) → ReLU → Linear(64, 32) → ReLU → Linear(32, n_actions) → Softmax
    ValueNet:  Linear(state_dim, 64) → ReLU → Linear(64, 32) → ReLU → Linear(32, 1)

Environment: CartPole-like balance task (pure Python, no gym needed)

Total: ~6K parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple


# ============================================================================
# Pure-Python CartPole Environment (no OpenAI Gym dependency)
# ============================================================================

class CartPoleLocal:
    """
    Classic CartPole environment — implemented from scratch.
    
    A pole is attached by an unactuated joint to a cart, which moves
    along a frictionless track. The system is controlled by applying a
    force of +1 or -1 to the cart. The goal is to prevent the pole from
    falling over.
    
    State: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    Actions: 0 (push left) or 1 (push right)
    
    Physics parameters match OpenAI Gym's CartPole-v1.
    """

    def __init__(self):
        # Physics constants
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5  # half-length of pole
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # time step

        # Episode termination thresholds
        self.x_threshold = 2.4
        self.theta_threshold = 12 * np.pi / 180  # 12 degrees

        self.state = None
        self.steps = 0
        self.max_steps = 500

    @property
    def state_dim(self):
        return 4

    @property
    def n_actions(self):
        return 2

    def reset(self, seed=None):
        """Reset to random initial state near equilibrium."""
        rng = np.random.RandomState(seed)
        self.state = rng.uniform(-0.05, 0.05, size=(4,)).astype(np.float32)
        self.steps = 0
        return self.state.copy()

    def step(self, action):
        """
        Apply action and simulate one timestep.
        Returns: (next_state, reward, done, info)
        """
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag

        # Physics simulation (Euler integration)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        temp = (force + self.polemass_length * theta_dot**2 * sin_theta) / self.total_mass
        theta_acc = (self.gravity * sin_theta - cos_theta * temp) / (
            self.length * (4.0/3.0 - self.masspole * cos_theta**2 / self.total_mass)
        )
        x_acc = temp - self.polemass_length * theta_acc * cos_theta / self.total_mass

        # Update state
        x += self.tau * x_dot
        x_dot += self.tau * x_acc
        theta += self.tau * theta_dot
        theta_dot += self.tau * theta_acc
        
        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self.steps += 1

        # Check termination
        done = bool(
            x < -self.x_threshold or x > self.x_threshold or
            theta < -self.theta_threshold or theta > self.theta_threshold or
            self.steps >= self.max_steps
        )

        reward = 1.0  # Reward every timestep (including terminal)
        return self.state.copy(), reward, done, {}


# ============================================================================
# Policy Gradient Agent
# ============================================================================

class PolicyNet(nn.Module):
    """
    Simple policy network that maps states to action probabilities.
    
    Students learn: this network outputs a probability distribution,
    not a deterministic action. The stochasticity enables exploration.
    """

    def __init__(self, state_dim=4, n_actions=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
            nn.Linear(hidden // 2, n_actions),
        )

    def forward(self, x):
        logits = self.net(x)
        return F.softmax(logits, dim=-1)


class ValueNet(nn.Module):
    """
    Value function (baseline) that estimates state value V(s).
    
    Subtracting V(s) from returns reduces variance in the
    policy gradient estimate (advantage = return - baseline).
    """

    def __init__(self, state_dim=4, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class REINFORCEAgent(nn.Module):
    """
    REINFORCE with baseline agent.
    
    Combines PolicyNet and ValueNet. The policy gradient is:
        ∇_θ J(θ) = E_τ [ Σ_t ∇_θ log π(a_t|s_t) * (G_t - V(s_t)) ]
    
    Args:
        state_dim: Observation space dimension (4 for CartPole)
        n_actions: Number of discrete actions (2 for CartPole)
        gamma: Discount factor
    """

    def __init__(self, state_dim=4, n_actions=2, gamma=0.99):
        super().__init__()
        self.policy = PolicyNet(state_dim, n_actions)
        self.value = ValueNet(state_dim)
        self.gamma = gamma

    def select_action(self, state):
        """Sample action from policy distribution."""
        state_t = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state_t)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), self.value(state_t)

    def compute_returns(self, rewards):
        """Compute discounted returns G_t = Σ_{k=0}^{T-t} γ^k r_{t+k}."""
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        # Normalize for stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns


# ============================================================================
# Training Loop
# ============================================================================

def train_rl_agent(n_episodes=300, lr=0.002, seed=42):
    """
    Train REINFORCE agent on local CartPole.
    
    Returns training metrics for convergence verification.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = CartPoleLocal()
    agent = REINFORCEAgent(state_dim=env.state_dim, n_actions=env.n_actions)
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)

    episode_rewards = []
    
    for ep in range(n_episodes):
        state = env.reset(seed=None)  # Random reset for exploration
        log_probs = []
        values = []
        rewards = []

        while True:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            
            state = next_state
            if done:
                break

        # Compute returns and advantages
        returns = agent.compute_returns(rewards)
        log_probs = torch.stack(log_probs)
        values = torch.stack(values).squeeze()
        
        # Policy loss: -log π(a|s) * advantage + entropy bonus
        advantages = returns - values.detach()
        policy_loss = -(log_probs * advantages).mean()
        
        # Entropy bonus for exploration
        probs_all = torch.stack([agent.policy(torch.FloatTensor(s).unsqueeze(0)) for s in [state]])
        entropy = -(probs_all * probs_all.log()).sum(-1).mean()
        
        # Value loss: MSE(V(s), G)
        value_loss = F.mse_loss(values, returns)
        
        # Combined loss with entropy bonus
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_rewards.append(sum(rewards))

        if (ep + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"  Episode {ep+1:4d}: avg_reward={avg_reward:.1f}  "
                  f"policy_loss={policy_loss.item():.4f}  "
                  f"value_loss={value_loss.item():.4f}")

    # Summary
    n_params = sum(p.numel() for p in agent.parameters())
    avg_final = np.mean(episode_rewards[-50:])
    
    return {
        "episode_rewards": episode_rewards,
        "avg_final_reward": avg_final,
        "n_params": n_params,
        "solved": avg_final >= 195,  # Classic CartPole solved threshold
    }


def get_rl_dataloaders(**kwargs):
    """
    RL doesn't use DataLoaders — returns an environment + agent factory.
    Compatible with dataset_factory interface.
    """
    return {
        "env": CartPoleLocal(),
        "agent_factory": lambda: REINFORCEAgent(),
        "type": "reinforcement_learning",
    }


if __name__ == "__main__":
    agent = REINFORCEAgent()
    n_params = sum(p.numel() for p in agent.parameters())
    print(f"REINFORCE Agent: {n_params:,} parameters")
    print()
    print("Training on local CartPole environment...")
    results = train_rl_agent(n_episodes=300)
    print(f"\n✅ Results:")
    print(f"   Final avg reward: {results['avg_final_reward']:.1f}")
    print(f"   Solved: {'Yes ✅' if results['solved'] else 'Not yet'}")
    print(f"   Parameters: {results['n_params']:,}")
