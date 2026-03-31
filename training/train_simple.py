"""
Simple Training Script for Grid Navigation (No Ray/RLlib Required)

This script implements a simple Q-learning / Policy Gradient approach
that doesn't require Ray/RLlib, making it suitable for quick experiments
and environments where Ray is not available.

Usage:
    python train_simple.py --episodes=1000 --render-freq=100

Features:
- Simple neural network policy (PyTorch)
- REINFORCE / Policy Gradient algorithm
- No external RL library dependencies (just PyTorch + Gymnasium)
- Checkpointing and logging
"""

import argparse
import os
import sys
import json
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import deque

import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Please install with: pip install torch")

# Try to import gymnasium
try:
    import gymnasium as gym
    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False
    print("Warning: Gymnasium not available. Please install with: pip install gymnasium")


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Environment
    world_size: float = 10.0
    num_static_obstacles: int = 3
    num_dynamic_obstacles: int = 1
    max_episode_steps: int = 500

    # Training
    num_episodes: int = 1000
    lr: float = 3e-4
    gamma: float = 0.99
    batch_size: int = 32

    # Network
    hidden_size: int = 256

    # Logging
    log_freq: int = 10
    save_freq: int = 100
    render_freq: int = 0  # 0 = don't render

    # Random seed
    seed: int = 42

    # Paths
    output_dir: str = "./training_output"

    def to_dict(self) -> Dict:
        return asdict(self)


class GridNavPolicy(nn.Module):
    """
    Simple neural network policy for grid navigation.

    Architecture:
    - Input: Dict observation (grid + scalar features)
    - CNN encoder for grid
    - MLP for combined features
    - Output: action mean and log_std for continuous actions
    """

    def __init__(self, action_dim: int = 3, hidden_size: int = 256):
        super().__init__()
        self.action_dim = action_dim

        # CNN for grid encoding (32x32 -> 256)
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
        self.combined = nn.Sequential(
            nn.Linear(combined_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Policy head
        self.policy_mean = nn.Linear(hidden_size, action_dim)
        self.policy_log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            obs: Dictionary with 'occupancy_grid', 'robot_pose', 'target_relative', 'velocity'

        Returns:
            action_mean: (batch, action_dim)
            action_std: (batch, action_dim)
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
        features = self.combined(combined)

        # Policy output
        action_mean = self.policy_mean(features)
        action_std = torch.exp(self.policy_log_std).expand_as(action_mean)

        return action_mean, action_std

    def select_action(self, obs: Dict[str, torch.Tensor], deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select action, return log probability and distribution."""
        mean, std = self.forward(obs)

        if deterministic:
            action = mean
            log_prob = torch.zeros(1)
            dist = None
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.rsample()  # Reparameterized sample for gradients
            log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob, dist


def train_policy_gradient(
    config: TrainingConfig,
    env,
    policy: GridNavPolicy,
    optimizer: torch.optim.Optimizer,
) -> Dict[str, List[float]]:
    """
    Train policy using REINFORCE (policy gradient).

    This is a simple baseline algorithm that works without complex RL libraries.

    Args:
        config: Training configuration
        env: Gymnasium environment
        policy: Neural network policy
        optimizer: Optimizer for policy parameters

    Returns:
        Dictionary containing training history
    """
    history = {
        'episode_rewards': [],
        'episode_lengths': [],
        'losses': [],
    }

    print(f"\n{'='*70}")
    print(f"Starting Training: {config.num_episodes} episodes")
    print(f"{'='*70}\n")

    for episode in range(1, config.num_episodes + 1):
        # Collect trajectory
        obs, info = env.reset(seed=config.seed + episode)

        log_probs = []
        rewards = []

        done = False
        step = 0
        episode_reward = 0.0

        while not done and step < config.max_episode_steps:
            # Convert obs to torch tensors
            obs_tensor = {k: torch.FloatTensor(v).unsqueeze(0) for k, v in obs.items()}

            # Select action (during data collection, no gradients needed)
            with torch.no_grad():
                action, _, _ = policy.select_action(obs_tensor, deterministic=False)
                action_np = action.squeeze(0).numpy()

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action_np)

            # Store trajectory
            rewards.append(reward)
            episode_reward += reward

            done = terminated or truncated
            obs = next_obs
            step += 1

        # Now re-compute log_probs with gradients for the collected trajectory
        # We need to re-run through the trajectory to get gradients
        obs, info = env.reset(seed=config.seed + episode)  # Reset to beginning
        step = 0
        done = False
        log_probs = []

        while not done and step < len(rewards):
            obs_tensor = {k: torch.FloatTensor(v).unsqueeze(0) for k, v in obs.items()}

            # Get action with gradients
            action, log_prob, _ = policy.select_action(obs_tensor, deterministic=False)
            action_np = action.detach().squeeze(0).numpy()

            log_probs.append(log_prob)

            # Step environment
            obs, _, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            step += 1

        # Compute returns (discounted cumulative rewards)
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + config.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns)
        # Normalize returns for stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute policy loss
        log_probs_tensor = torch.stack(log_probs)
        loss = -(log_probs_tensor * returns).mean()

        # Update policy
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        optimizer.step()

        # Record history
        history['episode_rewards'].append(episode_reward)
        history['episode_lengths'].append(step)
        history['losses'].append(loss.item())

        # Print progress
        if episode % config.log_freq == 0 or episode == 1:
            recent_rewards = history['episode_rewards'][-config.log_freq:]
            avg_reward = np.mean(recent_rewards)
            print(f"Episode {episode:5d}/{config.num_episodes} | "
                  f"Avg Reward: {avg_reward:8.2f} | "
                  f"Steps: {step:4d} | "
                  f"Loss: {loss.item():.4f}")

        # Save checkpoint
        if config.save_freq > 0 and episode % config.save_freq == 0:
            checkpoint_path = os.path.join(config.output_dir, f"checkpoint_{episode}.pt")
            torch.save({
                'episode': episode,
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
            }, checkpoint_path)
            print(f"  -> Saved checkpoint: {checkpoint_path}")

    # Save final model
    final_path = os.path.join(config.output_dir, "final_model.pt")
    torch.save({
        'episode': config.num_episodes,
        'policy_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
    }, final_path)
    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"Final model saved: {final_path}")
    print(f"{'='*70}\n")

    return history


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train grid navigation policy")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--hidden-size", type=int, default=256, help="Hidden layer size")
    parser.add_argument("--log-freq", type=int, default=10, help="Logging frequency")
    parser.add_argument("--save-freq", type=int, default=100, help="Checkpoint frequency")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="./training_output", help="Output directory")

    args = parser.parse_args()

    # Check dependencies
    if not TORCH_AVAILABLE:
        print("Error: PyTorch not available. Please install with: pip install torch")
        return 1

    if not GYMNASIUM_AVAILABLE:
        print("Error: Gymnasium not available. Please install with: pip install gymnasium")
        return 1

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create config
    config = TrainingConfig(
        num_episodes=args.episodes,
        lr=args.lr,
        gamma=args.gamma,
        hidden_size=args.hidden_size,
        log_freq=args.log_freq,
        save_freq=args.save_freq,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    # Save config
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"Config saved to: {config_path}")

    # Import and create environment
    from envs import OccupancyGridEnv

    env_config = {
        "world_width": 10.0,
        "world_height": 10.0,
        "num_static_obstacles": 3,
        "num_dynamic_obstacles": 1,
        "max_episode_steps": 500,
        "random_seed": config.seed,
    }

    env = OccupancyGridEnv(**env_config)

    # Create policy and optimizer
    policy = GridNavPolicy(action_dim=3, hidden_size=config.hidden_size)
    optimizer = optim.Adam(policy.parameters(), lr=config.lr)

    print(f"\nPolicy parameters: {sum(p.numel() for p in policy.parameters()):,}")

    # Train
    history = train_policy_gradient(config, env, policy, optimizer)

    # Plot training curve if matplotlib available
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Episode rewards
        ax1.plot(history['episode_rewards'], alpha=0.3)
        # Moving average
        window = min(50, len(history['episode_rewards']) // 10)
        if window > 1:
            ma = np.convolve(history['episode_rewards'], np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(history['episode_rewards'])), ma, 'r-', linewidth=2)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Training Reward')
        ax1.grid(True, alpha=0.3)

        # Losses
        ax2.plot(history['losses'])
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Loss')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(args.output_dir, "training_curves.png")
        plt.savefig(plot_path, dpi=150)
        print(f"\nTraining curves saved to: {plot_path}")

    except ImportError:
        print("\nMatplotlib not available, skipping training curve plot")

    return 0


if __name__ == "__main__":
    sys.exit(main())
