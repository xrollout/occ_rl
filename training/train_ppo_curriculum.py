"""
PPO Training with Curriculum Learning

This script implements curriculum learning for the grid navigation task.
The curriculum gradually increases difficulty:

Phase 1 (Easy): No obstacles, learn goal-seeking behavior
Phase 2 (Medium): 1-2 static obstacles, learn basic obstacle avoidance
Phase 3 (Hard): 3-5 static obstacles, complex navigation
Phase 4 (Expert): Full environment with dynamic obstacles

Each phase trains until performance threshold is met before advancing.
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs import OccupancyGridEnv
from training.train_ppo_custom import (
    PPOConfig, ActorCriticPolicy, compute_gae, collect_rollouts
)


class CurriculumPhase(NamedTuple):
    """Defines a curriculum phase with specific environment settings."""
    name: str
    num_static_obstacles: int
    num_dynamic_obstacles: int
    target_success_rate: float  # Success rate to advance to next phase
    min_episodes: int  # Minimum episodes before can advance
    max_episodes: int  # Maximum episodes in this phase


# Define curriculum phases
CURRICULUM_PHASES = [
    CurriculumPhase(
        name="Easy (No Obstacles)",
        num_static_obstacles=0,
        num_dynamic_obstacles=0,
        target_success_rate=0.80,  # 80% success to advance
        min_episodes=50,
        max_episodes=200,
    ),
    CurriculumPhase(
        name="Medium (1-2 Static Obstacles)",
        num_static_obstacles=2,
        num_dynamic_obstacles=0,
        target_success_rate=0.70,  # 70% success to advance
        min_episodes=100,
        max_episodes=300,
    ),
    CurriculumPhase(
        name="Hard (3-5 Static Obstacles)",
        num_static_obstacles=4,
        num_dynamic_obstacles=0,
        target_success_rate=0.60,  # 60% success to advance
        min_episodes=150,
        max_episodes=400,
    ),
    CurriculumPhase(
        name="Expert (Full Environment)",
        num_static_obstacles=5,
        num_dynamic_obstacles=2,
        target_success_rate=0.70,  # 70% success for production
        min_episodes=200,
        max_episodes=1000,  # Train extensively in final phase
    ),
]


class CurriculumMetrics:
    """Tracks metrics for curriculum learning."""

    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_outcomes = []  # 'success', 'collision', 'timeout'
        self.phase_history = []

    def add_episode(self, reward: float, length: int, outcome: str, phase: int):
        """Record an episode."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_outcomes.append(outcome)
        self.phase_history.append(phase)

    def get_recent_stats(self, n: int = 50) -> Dict:
        """Get statistics for recent n episodes."""
        if len(self.episode_rewards) < n:
            n = len(self.episode_rewards)

        recent_rewards = self.episode_rewards[-n:]
        recent_outcomes = self.episode_outcomes[-n:]

        success_count = sum(1 for o in recent_outcomes if o == 'success')
        collision_count = sum(1 for o in recent_outcomes if o == 'collision')
        timeout_count = sum(1 for o in recent_outcomes if o == 'timeout')

        return {
            'n_episodes': n,
            'mean_reward': np.mean(recent_rewards),
            'std_reward': np.std(recent_rewards),
            'success_rate': success_count / n,
            'collision_rate': collision_count / n,
            'timeout_rate': timeout_count / n,
            'success_count': success_count,
            'collision_count': collision_count,
            'timeout_count': timeout_count,
        }

    def get_phase_stats(self, phase: int) -> Dict:
        """Get statistics for a specific phase."""
        indices = [i for i, p in enumerate(self.phase_history) if p == phase]

        if not indices:
            return {'n_episodes': 0}

        phase_rewards = [self.episode_rewards[i] for i in indices]
        phase_outcomes = [self.episode_outcomes[i] for i in indices]

        success_count = sum(1 for o in phase_outcomes if o == 'success')

        return {
            'n_episodes': len(indices),
            'mean_reward': np.mean(phase_rewards),
            'std_reward': np.std(phase_rewards),
            'success_rate': success_count / len(indices),
            'success_count': success_count,
        }


def evaluate_policy(env, policy, num_episodes=50, seed=42):
    """Evaluate a policy and return statistics."""
    rewards = []
    outcomes = {'success': 0, 'collision': 0, 'timeout': 0}

    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep)
        episode_reward = 0
        steps = 0
        done = False

        while not done and steps < 500:
            obs_tensor = {k: torch.FloatTensor(v).unsqueeze(0) for k, v in obs.items()}

            with torch.no_grad():
                action, _, _ = policy.get_action(obs_tensor, deterministic=True)
                action_np = action[0]

            obs, reward, terminated, truncated, info = env.step(action_np)
            episode_reward += reward
            steps += 1
            done = terminated or truncated

        rewards.append(episode_reward)

        if info.get('goal_reached', False):
            outcomes['success'] += 1
        elif info.get('collision', False):
            outcomes['collision'] += 1
        else:
            outcomes['timeout'] += 1

    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards),
        'success_rate': outcomes['success'] / num_episodes,
        'collision_rate': outcomes['collision'] / num_episodes,
        'timeout_rate': outcomes['timeout'] / num_episodes,
        'outcomes': outcomes,
    }


def train_curriculum(config: PPOConfig):
    """Main curriculum learning training loop."""

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    os.makedirs(config.output_dir, exist_ok=True)

    # Set seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create policy
    policy = ActorCriticPolicy(action_dim=3, hidden_size=256).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=config.lr, eps=1e-5)

    # Initialize curriculum
    current_phase = 0
    phase_episodes = 0
    curriculum_metrics = CurriculumMetrics()

    # Training state
    total_timesteps = 0
    best_success_rate = 0.0

    print()
    print('=' * 90)
    print('STARTING PPO CURRICULUM LEARNING')
    print('=' * 90)
    print(f'Policy parameters: {sum(p.numel() for p in policy.parameters()):,}')
    print(f'Total timesteps: {config.total_timesteps:,}')
    print(f'Curriculum phases: {len(CURRICULUM_PHASES)}')
    print('=' * 90)
    print()

    # Curriculum training loop
    while current_phase < len(CURRICULUM_PHASES) and total_timesteps < config.total_timesteps:
        phase_config = CURRICULUM_PHASES[current_phase]

        print(f'\n{"="*90}')
        print(f'CURRICULUM PHASE {current_phase + 1}/{len(CURRICULUM_PHASES)}: {phase_config.name}')
        print(f'{"="*90}')
        print(f'Environment: {phase_config.num_static_obstacles} static, {phase_config.num_dynamic_obstacles} dynamic obstacles')
        print(f'Target success rate: {phase_config.target_success_rate*100:.0f}%')
        print(f'Episode range: {phase_config.min_episodes} - {phase_config.max_episodes}')
        print()

        # Create environment for this phase
        env = OccupancyGridEnv(
            world_width=config.world_size,
            world_height=config.world_size,
            num_static_obstacles=phase_config.num_static_obstacles,
            num_dynamic_obstacles=phase_config.num_dynamic_obstacles,
            max_episode_steps=config.max_episode_steps,
            random_seed=config.seed + current_phase * 1000,
        )

        # Training loop for this phase
        phase_episodes = 0
        phase_successes = 0
        phase_rewards = []

        while phase_episodes < phase_config.max_episodes and total_timesteps < config.total_timesteps:
            # Collect rollouts
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

            # Update policy
            n_samples = len(rollout_data['rewards'])
            indices = np.arange(n_samples)

            for epoch in range(config.n_epochs):
                np.random.shuffle(indices)

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

            # Update counters
            n_episodes = len(rollout_data['rewards'])
            total_timesteps += n_episodes
            phase_episodes += 1

            # Track outcomes
            for i in range(len(rollout_data['rewards'])):
                reward = rollout_data['rewards'][i].item()
                phase_rewards.append(reward)

                # Determine outcome (simplified)
                if reward > 100:  # High reward likely means success
                    phase_successes += 1

            # Log progress
            if phase_episodes % 10 == 0:
                recent_successes = sum(1 for r in phase_rewards[-50:] if r > 100)
                recent_rate = recent_successes / min(len(phase_rewards[-50:]), 50) if phase_rewards else 0
                print(f'  Phase {current_phase+1} Episode {phase_episodes}: '
                      f'Avg Reward={np.mean(phase_rewards[-50:]):.1f}, '
                      f'Success Rate={recent_rate*100:.1f}%')

            # Check if ready to advance
            if phase_episodes >= phase_config.min_episodes:
                recent_episodes = min(50, len(phase_rewards))
                recent_successes = sum(1 for r in phase_rewards[-recent_episodes:] if r > 100)
                recent_success_rate = recent_successes / recent_episodes if recent_episodes > 0 else 0

                if recent_success_rate >= phase_config.target_success_rate:
                    print(f'\n✓ Phase {current_phase+1} complete! Success rate: {recent_success_rate*100:.1f}%')
                    break

            # Check if max episodes reached
            if phase_episodes >= phase_config.max_episodes:
                print(f'\n⚠ Phase {current_phase+1} max episodes reached. Moving to next phase.')
                break

        # Advance to next phase
        current_phase += 1

        # Save checkpoint after each phase
        checkpoint_path = os.path.join(config.output_dir, f'checkpoint_phase_{current_phase}.pt')
        torch.save({
            'phase': current_phase,
            'policy_state_dict': policy.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'total_timesteps': total_timesteps,
        }, checkpoint_path)
        print(f'\nSaved checkpoint: {checkpoint_path}')

    # Training complete
    print()
    print('=' * 90)
    print('CURRICULUM LEARNING COMPLETE')
    print('=' * 90)
    print(f'Total timesteps: {total_timesteps:,}')
    print(f'Final phase: {current_phase}/{len(CURRICULUM_PHASES)}')

    # Save final model
    final_path = os.path.join(config.output_dir, 'final_model.pt')
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'total_timesteps': total_timesteps,
    }, final_path)
    print(f'\nSaved final model: {final_path}')


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train PPO with curriculum learning')
    parser.add_argument('--timesteps-per-phase', type=int, default=100000,
                        help='Timesteps per curriculum phase')
    parser.add_argument('--n-steps', type=int, default=2048, help='Steps per update')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', type=str, default='./ppo_curriculum_output',
                        help='Output directory')

    args = parser.parse_args()

    # Create config
    config = PPOConfig(
        total_timesteps=args.timesteps_per_phase * len(CURRICULUM_PHASES),
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    # Train
    train_curriculum(config)


if __name__ == '__main__':
    main()
