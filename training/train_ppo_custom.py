"""
PPO Training Implementation (Pure PyTorch)

This script implements PPO (Proximal Policy Optimization) from scratch
using only PyTorch, without requiring RLlib.

Key features:
- Clipped surrogate objective
- Value function baseline
- Generalized Advantage Estimation (GAE)
- Mini-batch updates
- Entropy regularization
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs import OccupancyGridEnv


@dataclass
class PPOConfig:
    """Configuration for PPO training."""
    # Environment
    world_size: float = 10.0
    num_static_obstacles: int = 3
    num_dynamic_obstacles: int = 1
    max_episode_steps: int = 500

    # Training
    total_timesteps: int = 500000
    n_steps: int = 2048  # Steps per update
    batch_size: int = 64
    n_epochs: int = 10  # Training epochs per update

    # PPO hyperparameters
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5

    # Logging
    log_freq: int = 10
    save_freq: int = 100000
    seed: int = 42
    output_dir: str = "./ppo_training_output"


class ActorCriticPolicy(nn.Module):
    """
    Actor-Critic policy network with CNN+MLP architecture.

    Outputs:
    - Policy: mean and log_std for continuous actions
    - Value: state value estimate
    """

    def __init__(self, action_dim: int = 3, hidden_size: int = 256):
        super().__init__()
        self.action_dim = action_dim

        # CNN encoder for grid (32x32 -> 256)
        self.grid_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
        )

        # MLP for scalar features (8 -> 64)
        self.scalar_encoder = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
        )

        # Combined processing (256 + 64 -> hidden_size)
        combined_dim = 256 + 64
        self.shared = nn.Sequential(
            nn.Linear(combined_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Actor head (policy)
        self.actor_mean = nn.Linear(hidden_size, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic head (value function)
        self.critic = nn.Linear(hidden_size, 1)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            obs: Dictionary with 'occupancy_grid', 'robot_pose', 'target_relative', 'velocity'

        Returns:
            action_mean: Mean of action distribution
            action_std: Std of action distribution
            value: State value estimate
        """
        # Encode grid
        grid = obs['occupancy_grid'].unsqueeze(1)  # Add channel dim
        grid_features = self.grid_encoder(grid)

        # Encode scalar features
        scalar_input = torch.cat([
            obs['robot_pose'],
            obs['target_relative'],
            obs['velocity'],
        ], dim=-1)
        scalar_features = self.scalar_encoder(scalar_input)

        # Combine
        combined = torch.cat([grid_features, scalar_features], dim=-1)
        shared_features = self.shared(combined)

        # Actor output
        action_mean = self.actor_mean(shared_features)
        action_std = torch.exp(self.actor_log_std).expand_as(action_mean)

        # Critic output
        value = self.critic(shared_features)

        return action_mean, action_std, value

    def get_action(self, obs: Dict[str, torch.Tensor], deterministic: bool = False) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """Get action from policy."""
        with torch.no_grad():
            mean, std, value = self.forward(obs)

            if deterministic:
                action = mean
            else:
                dist = Normal(mean, std)
                action = dist.sample()

            log_prob = Normal(mean, std).log_prob(action).sum(dim=-1)

        return action.cpu().numpy(), log_prob, value.squeeze(-1)


def compute_gae(rewards: torch.Tensor,
                values: torch.Tensor,
                dones: torch.Tensor,
                gamma: float = 0.99,
                gae_lambda: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards: Reward tensor (n_steps,)
        values: Value estimates (n_steps + 1,)
        dones: Done flags (n_steps,)
        gamma: Discount factor
        gae_lambda: GAE lambda parameter

    Returns:
        advantages: Advantage estimates
        returns: Target values for value function
    """
    n_steps = len(rewards)
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0

    # Compute advantages backwards
    for t in reversed(range(n_steps)):
        if t == n_steps - 1:
            next_value = values[-1]
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
        advantages[t] = last_gae

    returns = advantages + values[:-1]

    return advantages, returns


def collect_rollouts(env: OccupancyGridEnv,
                     policy: ActorCriticPolicy,
                     n_steps: int,
                     device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Collect rollout data from environment.

    Returns:
        Dictionary with observations, actions, rewards, etc.
    """
    obs_list = {key: [] for key in ['occupancy_grid', 'robot_pose', 'target_relative', 'velocity']}
    actions_list = []
    rewards_list = []
    dones_list = []
    values_list = []
    log_probs_list = []

    obs, _ = env.reset()

    for step in range(n_steps):
        # Convert obs to tensor
        obs_tensor = {k: torch.FloatTensor(v).unsqueeze(0).to(device) for k, v in obs.items()}

        # Get action from policy
        action, log_prob, value = policy.get_action(obs_tensor, deterministic=False)

        # Store data
        for key in obs_list:
            obs_list[key].append(obs[key].copy())
        actions_list.append(action[0])
        log_probs_list.append(log_prob.item())
        values_list.append(value.item())

        # Step environment
        obs, reward, terminated, truncated, _ = env.step(action[0])

        rewards_list.append(reward)
        done = terminated or truncated
        dones_list.append(float(done))

        if done:
            obs, _ = env.reset()

    # Get final value for bootstrap
    obs_tensor = {k: torch.FloatTensor(v).unsqueeze(0).to(device) for k, v in obs.items()}
    with torch.no_grad():
        _, _, final_value = policy.get_action(obs_tensor)
    values_list.append(final_value.item())

    # Convert to tensors
    data = {
        'obs': {k: torch.FloatTensor(np.array(v)) for k, v in obs_list.items()},
        'actions': torch.FloatTensor(np.array(actions_list)),
        'rewards': torch.FloatTensor(rewards_list),
        'dones': torch.FloatTensor(dones_list),
        'values': torch.FloatTensor(values_list),
        'log_probs': torch.FloatTensor(log_probs_list),
    }

    return data


def train_ppo(config: PPOConfig, pretrained_ckpt_path: Optional[str] = None):
    """Main PPO training loop.

    Args:
        config: PPO configuration.
        pretrained_ckpt_path: Path to pretrained checkpoint (from behavior cloning).
            If provided, loads policy weights before training.
    """

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    os.makedirs(config.output_dir, exist_ok=True)

    # Set seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create environment
    env = OccupancyGridEnv(
        world_width=config.world_size,
        world_height=config.world_size,
        num_static_obstacles=config.num_static_obstacles,
        num_dynamic_obstacles=config.num_dynamic_obstacles,
        max_episode_steps=config.max_episode_steps,
        random_seed=config.seed,
    )

    # Create policy
    policy = ActorCriticPolicy(action_dim=3, hidden_size=256).to(device)

    # Load pretrained weights if provided
    if pretrained_ckpt_path is not None:
        print(f"Loading pretrained checkpoint from: {pretrained_ckpt_path}")
        ckpt = torch.load(pretrained_ckpt_path, map_location=device, weights_only=False)
        policy.load_state_dict(ckpt['policy_state_dict'])
        print(f"Successfully loaded pretrained weights")

    # Create optimizer
    optimizer = optim.Adam(policy.parameters(), lr=config.lr, eps=1e-5)

    # Training metrics
    episode_rewards = []
    episode_lengths = []
    value_losses = []
    policy_losses = []
    entropies = []

    total_timesteps = 0
    episode_num = 0

    print()
    print('=' * 80)
    print('STARTING PPO TRAINING')
    print('=' * 80)
    print(f'Policy parameters: {sum(p.numel() for p in policy.parameters()):,}')
    print(f'Total timesteps: {config.total_timesteps:,}')
    print(f'Update frequency: {config.n_steps} steps')
    print(f'Batch size: {config.batch_size}')
    print(f'Learning rate: {config.lr}')
    print('=' * 80)
    print()

    # Training loop
    while total_timesteps < config.total_timesteps:
        # Collect rollouts
        print(f'Collecting rollouts... (timestep {total_timesteps}/{config.total_timesteps})')
        rollout_data = collect_rollouts(env, policy, config.n_steps, device)

        # Compute advantages
        advantages, returns = compute_gae(
            rollout_data['rewards'],
            rollout_data['values'],
            rollout_data['dones'],
            config.gamma,
            config.gae_lambda
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Training
        n_samples = len(rollout_data['rewards'])
        indices = np.arange(n_samples)

        epoch_policy_losses = []
        epoch_value_losses = []
        epoch_entropies = []

        for epoch in range(config.n_epochs):
            # Shuffle indices
            np.random.shuffle(indices)

            # Mini-batch updates
            for start in range(0, n_samples, config.batch_size):
                end = min(start + config.batch_size, n_samples)
                mb_indices = indices[start:end]

                # Get mini-batch data
                mb_obs = {k: v[mb_indices].to(device) for k, v in rollout_data['obs'].items()}
                mb_actions = rollout_data['actions'][mb_indices].to(device)
                mb_old_log_probs = rollout_data['log_probs'][mb_indices].to(device)
                mb_advantages = advantages[mb_indices].to(device)
                mb_returns = returns[mb_indices].to(device)

                # Forward pass
                mean, std, values = policy(mb_obs)
                dist = Normal(mean, std)
                log_probs = dist.log_prob(mb_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                # PPO loss
                ratio = torch.exp(log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - config.clip_range, 1 + config.clip_range) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_pred_clipped = rollout_data['values'][mb_indices].to(device) + \
                    torch.clamp(values.squeeze(-1) - rollout_data['values'][mb_indices].to(device),
                               -config.clip_range, config.clip_range)
                value_loss1 = (values.squeeze(-1) - mb_returns).pow(2)
                value_loss2 = (value_pred_clipped - mb_returns).pow(2)
                value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()

                # Total loss
                loss = policy_loss + config.vf_coef * value_loss - config.ent_coef * entropy

                # Optimization step
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
                optimizer.step()

                # Record metrics
                epoch_policy_losses.append(policy_loss.item())
                epoch_value_losses.append(value_loss.item())
                epoch_entropies.append(entropy.item())

        # Update counters
        total_timesteps += n_samples

        # Record metrics
        if epoch_policy_losses:
            policy_losses.append(np.mean(epoch_policy_losses))
            value_losses.append(np.mean(epoch_value_losses))
            entropies.append(np.mean(epoch_entropies))

        # Logging
        if len(policy_losses) % config.log_freq == 0:
            print(f'Update {len(policy_losses)} | '
                  f'Policy Loss: {policy_losses[-1]:.4f} | '
                  f'Value Loss: {value_losses[-1]:.4f} | '
                  f'Entropy: {entropies[-1]:.4f}')

        # Save checkpoint
        if total_timesteps % config.save_freq < config.n_steps:
            checkpoint_path = os.path.join(config.output_dir, f'checkpoint_{total_timesteps}.pt')
            torch.save({
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'total_timesteps': total_timesteps,
                'policy_losses': policy_losses,
                'value_losses': value_losses,
                'entropies': entropies,
            }, checkpoint_path)
            print(f'Saved checkpoint: {checkpoint_path}')

    # Save final model
    final_path = os.path.join(config.output_dir, 'final_model.pt')
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'total_timesteps': total_timesteps,
        'policy_losses': policy_losses,
        'value_losses': value_losses,
        'entropies': entropies,
    }, final_path)
    print(f'Saved final model: {final_path}')

    print()
    print('=' * 80)
    print('TRAINING COMPLETE')
    print('=' * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train PPO policy')
    parser.add_argument('--timesteps', type=int, default=500000, help='Total timesteps')
    parser.add_argument('--n-steps', type=int, default=2048, help='Steps per update')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--clip-range', type=float, default=0.2, help='PPO clip range')
    parser.add_argument('--n-epochs', type=int, default=10, help='Training epochs per update')
    parser.add_argument('--ent-coef', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--vf-coef', type=float, default=0.5, help='Value function coefficient')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', type=str, default='./ppo_training_output', help='Output directory')
    parser.add_argument('--pretrained-ckpt', type=str, default=None,
                        help='Path to pretrained checkpoint (from behavior cloning) to initialize from')

    args = parser.parse_args()

    # Create config
    config = PPOConfig(
        total_timesteps=args.timesteps,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        n_epochs=args.n_epochs,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    # Train
    train_ppo(config, pretrained_ckpt_path=args.pretrained_ckpt)


if __name__ == '__main__':
    main()
