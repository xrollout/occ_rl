"""
Behavior Cloning (and DAgger) Training from LLM Demonstrations

This script trains the policy via behavior cloning on the LLM demonstration
dataset. It also supports DAgger (Dataset Aggregation) for iterative
dataset improvement.
"""

import argparse
import os
import sys
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs import OccupancyGridEnv
from training.train_ppo_custom import ActorCriticPolicy
from llm_teacher.llm_teacher import LLMTeacher


@dataclass
class BCConfig:
    """Configuration for behavior cloning training."""
    # Dataset
    dataset_path: str
    train_split: float = 0.9

    # Training
    batch_size: int = 64
    lr: float = 3e-4
    epochs: int = 50
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # DAgger (iterative dataset aggregation)
    use_dagger: bool = False
    dagger_iterations: int = 5
    dagger_episodes_per_iter: int = 20

    # Output
    output_dir: str = "./bc_training_output"
    seed: int = 42

    # Architecture
    hidden_size: int = 256


class DemonstrationDataset(Dataset):
    """PyTorch Dataset for LLM demonstration transitions."""

    def __init__(self, dataset_path: str):
        """
        Load dataset from .npz file.

        Args:
            dataset_path: Path to .npz file with demonstration data.
        """
        data = np.load(dataset_path, allow_pickle=True)

        self.obs_grid = torch.FloatTensor(data['obs_grid'])
        self.obs_robot_pose = torch.FloatTensor(data['obs_robot_pose'])
        self.obs_target_relative = torch.FloatTensor(data['obs_target_relative'])
        self.obs_velocity = torch.FloatTensor(data['obs_velocity'])
        self.actions = torch.FloatTensor(data['actions'])
        self.rewards = torch.FloatTensor(data['rewards'])
        self.dones = torch.FloatTensor(data['dones'])

        # Precompute reward-to-go for value function training
        self.reward_to_go = self._compute_reward_togo(self.rewards, self.dones)

        print(f"Loaded dataset: {len(self)} transitions")

    def _compute_reward_togo(self, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """Compute reward-to-go (discounted sum of future rewards) from trajectory."""
        n = len(rewards)
        rtg = torch.zeros_like(rewards)
        future_reward = 0.0
        gamma = 0.99

        # Traverse backwards
        for i in reversed(range(n)):
            if dones[i]:
                future_reward = 0.0
            future_reward = rewards[i] + gamma * future_reward
            rtg[i] = future_reward

        return rtg

    def __len__(self) -> int:
        return len(self.obs_grid)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'occupancy_grid': self.obs_grid[idx],
            'robot_pose': self.obs_robot_pose[idx],
            'target_relative': self.obs_target_relative[idx],
            'velocity': self.obs_velocity[idx],
            'action': self.actions[idx],
            'reward_to_go': self.reward_to_go[idx],
        }


def add_dagger_data(
    policy: ActorCriticPolicy,
    llm_teacher: LLMTeacher,
    current_dataset: DemonstrationDataset,
    num_episodes: int,
    world_size: float,
    num_static_obstacles: int,
    num_dynamic_obstacles: int,
    max_episode_steps: int,
    device: torch.device,
) -> DemonstrationDataset:
    """
    Add new aggregated data using DAgger:
    - Run current policy in environment
    - For each state encountered, ask LLM teacher for correct action
    - Add (state, LLM action) to dataset

    Args:
        policy: Current policy.
        llm_teacher: LLM teacher.
        current_dataset: Current dataset.
        num_episodes: Number of new episodes to collect.
        world_size: World size.
        num_static_obstacles: Number of static obstacles.
        num_dynamic_obstacles: Number of dynamic obstacles.
        max_episode_steps: Max steps per episode.
        device: Torch device.

    Returns:
        Updated dataset with new examples.
    """
    print(f"\n[DAgger] Collecting {num_episodes} new episodes...")

    env = OccupancyGridEnv(
        world_width=world_size,
        world_height=world_size,
        num_static_obstacles=num_static_obstacles,
        num_dynamic_obstacles=num_dynamic_obstacles,
        max_episode_steps=max_episode_steps,
    )

    # New data storage
    new_obs_grid = []
    new_obs_robot_pose = []
    new_obs_target_relative = []
    new_obs_velocity = []
    new_actions = []
    new_rewards = []
    new_dones = []

    successes = 0

    for episode in tqdm(range(num_episodes)):
        obs, _ = env.reset()
        done = False
        steps = 0

        while not done and steps < max_episode_steps:
            # Get LLM action for this state (teacher correction)
            llm_action = llm_teacher.get_action(obs, use_cache=True)

            # Store the observation with LLM action
            new_obs_grid.append(obs['occupancy_grid'].copy())
            new_obs_robot_pose.append(obs['robot_pose'].copy())
            new_obs_target_relative.append(obs['target_relative'].copy())
            new_obs_velocity.append(obs['velocity'].copy())
            new_actions.append(llm_action.copy())

            # Step with current policy (we just collect states visited by policy)
            obs_tensor = {k: torch.FloatTensor(v).unsqueeze(0).to(device) for k, v in obs.items()}
            policy_action, _, _ = policy.get_action(obs_tensor, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(policy_action[0])

            done = terminated or truncated
            new_rewards.append(reward)
            new_dones.append(float(done))
            steps += 1

            if info.get('success', False):
                successes += 1

    print(f"[DAgger] Collected {len(new_obs_grid)} new transitions, {successes} successes")

    # Concatenate with existing data
    new_obs_grid = torch.FloatTensor(np.array(new_obs_grid))
    new_obs_robot_pose = torch.FloatTensor(np.array(new_obs_robot_pose))
    new_obs_target_relative = torch.FloatTensor(np.array(new_obs_target_relative))
    new_obs_velocity = torch.FloatTensor(np.array(new_obs_velocity))
    new_actions = torch.FloatTensor(np.array(new_actions))
    new_rewards = torch.FloatTensor(np.array(new_rewards))
    new_dones = torch.FloatTensor(np.array(new_dones))

    # Append to current dataset
    current_dataset.obs_grid = torch.cat([current_dataset.obs_grid, new_obs_grid], dim=0)
    current_dataset.obs_robot_pose = torch.cat([current_dataset.obs_robot_pose, new_obs_robot_pose], dim=0)
    current_dataset.obs_target_relative = torch.cat([current_dataset.obs_target_relative, new_obs_target_relative], dim=0)
    current_dataset.obs_velocity = torch.cat([current_dataset.obs_velocity, new_obs_velocity], dim=0)
    current_dataset.actions = torch.cat([current_dataset.actions, new_actions], dim=0)
    current_dataset.rewards = torch.cat([current_dataset.rewards, new_rewards], dim=0)
    current_dataset.dones = torch.cat([current_dataset.dones, new_dones], dim=0)
    current_dataset.reward_to_go = current_dataset._compute_reward_togo(current_dataset.rewards, current_dataset.dones)

    print(f"[DAgger] Dataset now has {len(current_dataset)} total transitions")
    return current_dataset


def train_bc(
    model: ActorCriticPolicy,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: BCConfig,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train behavior cloning for one epoch.

    Args:
        model: ActorCriticPolicy model.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        config: BC configuration.
        device: Torch device.

    Returns:
        (avg_train_loss, avg_val_loss)
    """
    optimizer = optim.Adam(model.parameters(), lr=config.lr, eps=1e-5)
    mse_loss = nn.MSELoss()

    # Training
    model.train()
    train_losses = []
    action_losses = []
    value_losses = []
    entropy_losses = []

    for batch in tqdm(train_loader, desc="Training"):
        # Move batch to device
        obs = {
            'occupancy_grid': batch['occupancy_grid'].to(device),
            'robot_pose': batch['robot_pose'].to(device),
            'target_relative': batch['target_relative'].to(device),
            'velocity': batch['velocity'].to(device),
        }
        target_action = batch['action'].to(device)
        target_rtg = batch['reward_to_go'].to(device)

        # Forward pass
        action_mean, action_std, value = model(obs)

        # MSE loss on action mean
        action_loss = mse_loss(action_mean, target_action)

        # MSE loss on value
        value_loss = mse_loss(value.squeeze(-1), target_rtg)

        # Entropy regularization (keep exploration)
        dist = torch.distributions.Normal(action_mean, action_std)
        entropy = dist.entropy().sum(dim=-1).mean()
        entropy_loss = -config.ent_coef * entropy

        # Total loss
        total_loss = action_loss + config.vf_coef * value_loss + entropy_loss

        # Optimization step
        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()

        train_losses.append(total_loss.item())
        action_losses.append(action_loss.item())
        value_losses.append(value_loss.item())
        entropy_losses.append(entropy_loss.item())

    avg_train_loss = np.mean(train_losses)

    # Validation
    model.eval()
    val_losses = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            obs = {
                'occupancy_grid': batch['occupancy_grid'].to(device),
                'robot_pose': batch['robot_pose'].to(device),
                'target_relative': batch['target_relative'].to(device),
                'velocity': batch['velocity'].to(device),
            }
            target_action = batch['action'].to(device)
            target_rtg = batch['reward_to_go'].to(device)

            action_mean, action_std, value = model(obs)

            action_loss = mse_loss(action_mean, target_action)
            value_loss = mse_loss(value.squeeze(-1), target_rtg)
            dist = torch.distributions.Normal(action_mean, action_std)
            entropy = dist.entropy().sum(dim=-1).mean()
            entropy_loss = -config.ent_coef * entropy

            total_loss = action_loss + config.vf_coef * value_loss + entropy_loss
            val_losses.append(total_loss.item())

    avg_val_loss = np.mean(val_losses)

    return avg_train_loss, avg_val_loss


def main():
    parser = argparse.ArgumentParser(description='Train behavior cloning from LLM demonstrations')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to demonstration dataset .npz file')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                        help='Entropy regularization coefficient')
    parser.add_argument('--vf-coef', type=float, default=0.5,
                        help='Value function coefficient')
    parser.add_argument('--train-split', type=float, default=0.9,
                        help='Train/validation split')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output-dir', type=str, default='./bc_training_output',
                        help='Output directory for checkpoints')
    parser.add_argument('--use-dagger', action='store_true',
                        help='Enable DAgger (iterative dataset aggregation)')
    parser.add_argument('--dagger-iterations', type=int, default=5,
                        help='Number of DAgger iterations')
    parser.add_argument('--dagger-episodes', type=int, default=20,
                        help='Number of episodes per DAgger iteration')
    parser.add_argument('--world-size', type=float, default=10.0,
                        help='World size for DAgger data collection')
    parser.add_argument('--num-static-obstacles', type=int, default=5,
                        help='Number of static obstacles for DAgger')
    parser.add_argument('--num-dynamic-obstacles', type=int, default=0,
                        help='Number of dynamic obstacles for DAgger')
    parser.add_argument('--max-episode-steps', type=int, default=200,
                        help='Maximum steps per episode for DAgger')

    args = parser.parse_args()

    # Create config
    config = BCConfig(
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        train_split=args.train_split,
        seed=args.seed,
        output_dir=args.output_dir,
        use_dagger=args.use_dagger,
        dagger_iterations=args.dagger_iterations,
    )

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    os.makedirs(config.output_dir, exist_ok=True)

    # Set seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Load dataset
    print(f"Loading dataset from: {config.dataset_path}")
    full_dataset = DemonstrationDataset(config.dataset_path)

    # Split into train/validation
    train_size = int(config.train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed),
    )

    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Create policy (same architecture as PPO)
    policy = ActorCriticPolicy(action_dim=3, hidden_size=config.hidden_size).to(device)

    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")

    # Training history
    train_losses = []
    val_losses = []

    print()
    print("=" * 80)
    print("STARTING BEHAVIOR CLONING TRAINING")
    print("=" * 80)
    print()

    # If using DAgger, we do multiple rounds of training + data collection
    if config.use_dagger:
        print(f"DAgger enabled: {config.dagger_iterations} iterations")
        llm_teacher = LLMTeacher()

        for dagger_iter in range(config.dagger_iterations):
            print(f"\n[DAgger] Iteration {dagger_iter + 1}/{config.dagger_iterations}")

            # Train on current dataset
            for epoch in range(config.epochs // config.dagger_iterations):
                avg_train, avg_val = train_bc(policy, train_loader, val_loader, config, device)
                train_losses.append(avg_train)
                val_losses.append(avg_val)
                print(f"  Epoch {len(train_losses)}: train_loss={avg_train:.6f}, val_loss={avg_val:.6f}")

            # Collect more data with current policy and LLM corrections
            if dagger_iter < config.dagger_iterations - 1:
                full_dataset = add_dagger_data(
                    policy,
                    llm_teacher,
                    full_dataset,
                    args.dagger_episodes,
                    args.world_size,
                    args.num_static_obstacles,
                    args.num_dynamic_obstacles,
                    args.max_episode_steps,
                    device,
                )

                # Resplit after adding data
                train_size = int(config.train_split * len(full_dataset))
                val_size = len(full_dataset) - train_size
                train_dataset, val_dataset = random_split(
                    full_dataset,
                    [train_size, val_size],
                    generator=torch.Generator().manual_seed(config.seed + dagger_iter),
                )

                train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    else:
        # Standard behavior cloning - just train once
        for epoch in range(config.epochs):
            avg_train, avg_val = train_bc(policy, train_loader, val_loader, config, device)
            train_losses.append(avg_train)
            val_losses.append(avg_val)
            print(f"Epoch {epoch + 1}/{config.epochs}: train_loss={avg_train:.6f}, val_loss={avg_val:.6f}")

    # Save final model
    final_path = os.path.join(config.output_dir, 'final_model.pt')
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'config': config.__dict__,
    }, final_path)

    print()
    print("=" * 80)
    print(f"TRAINING COMPLETE")
    print(f"Final model saved: {final_path}")
    print(f"Final train loss: {train_losses[-1]:.6f}")
    print(f"Final validation loss: {val_losses[-1]:.6f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
