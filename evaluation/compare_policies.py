"""
Compare Policies: Baseline PPO vs LLM-Imitation + PPO

This script evaluates two policies (baseline vs proposed) on the same set
of evaluation episodes and compares performance metrics.

Metrics compared:
- Success rate
- Collision rate
- Average episode length
- Average episode reward
- Standard deviation of reward
- Distance traveled
"""

import argparse
import os
import sys
import json
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple

import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs import OccupancyGridEnv
from training.train_ppo_custom import ActorCriticPolicy


@dataclass
class EvaluationResult:
    """Result from evaluating a policy."""
    num_episodes: int
    num_successes: int
    num_collisions: int
    success_rate: float
    collision_rate: float
    avg_episode_length: float
    avg_episode_reward: float
    std_episode_reward: float
    avg_distance_traveled: float
    all_rewards: List[float]
    all_lengths: List[float]


def evaluate_policy(
    policy: ActorCriticPolicy,
    num_episodes: int,
    world_size: float,
    num_static_obstacles: int,
    num_dynamic_obstacles: int,
    max_episode_steps: int,
    device: torch.device,
    seed_offset: int = 0,
) -> EvaluationResult:
    """
    Evaluate policy on multiple episodes with fixed seeds.

    Args:
        policy: Policy to evaluate.
        num_episodes: Number of evaluation episodes.
        world_size: World size.
        num_static_obstacles: Number of static obstacles.
        num_dynamic_obstacles: Number of dynamic obstacles.
        max_episode_steps: Maximum steps per episode.
        device: Torch device.
        seed_offset: Offset added to seed (for reproducibility).

    Returns:
        EvaluationResult with metrics.
    """
    policy.eval()

    num_successes = 0
    num_collisions = 0
    all_rewards = []
    all_lengths = []
    all_distances = []

    for episode_idx in tqdm(range(num_episodes), desc="Evaluating"):
        seed = episode_idx + seed_offset
        env = OccupancyGridEnv(
            world_width=world_size,
            world_height=world_size,
            num_static_obstacles=num_static_obstacles,
            num_dynamic_obstacles=num_dynamic_obstacles,
            max_episode_steps=max_episode_steps,
            random_seed=seed,
        )

        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        episode_steps = 0
        start_position = info['robot_position'].copy()
        prev_position = start_position.copy()
        distance_traveled = 0.0
        success = False
        collision = False

        with torch.no_grad():
            while not done and episode_steps < max_episode_steps:
                obs_tensor = {
                    k: torch.FloatTensor(v).unsqueeze(0).to(device)
                    for k, v in obs.items()
                }
                action, _, _ = policy.get_action(obs_tensor, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action[0])
                done = terminated or truncated
                episode_reward += reward
                episode_steps += 1

                # Track distance
                if 'robot_position' in info:
                    curr_position = np.array(info['robot_position'])
                else:
                    # Extract from obs['robot_pose']
                    curr_position = np.array(obs['robot_pose'][:2])
                distance_traveled += np.linalg.norm(curr_position - prev_position)
                prev_position = curr_position

                if info.get('goal_reached', False):
                    success = True
                    num_successes += 1
                if info.get('collision', False):
                    collision = True
                    num_collisions += 1

        all_rewards.append(episode_reward)
        all_lengths.append(episode_steps)
        all_distances.append(distance_traveled)

    # Compute aggregated metrics
    avg_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    avg_length = np.mean(all_lengths)
    avg_distance = np.mean(all_distances)
    success_rate = num_successes / num_episodes
    collision_rate = num_collisions / num_episodes

    return EvaluationResult(
        num_episodes=num_episodes,
        num_successes=num_successes,
        num_collisions=num_collisions,
        success_rate=success_rate,
        collision_rate=collision_rate,
        avg_episode_length=avg_length,
        avg_episode_reward=avg_reward,
        std_episode_reward=std_reward,
        avg_distance_traveled=avg_distance,
        all_rewards=all_rewards,
        all_lengths=all_lengths,
    )


def load_policy(checkpoint_path: str, device: torch.device) -> ActorCriticPolicy:
    """Load policy from checkpoint."""
    policy = ActorCriticPolicy(action_dim=3, hidden_size=256).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt['policy_state_dict'])
    print(f"Loaded policy from: {checkpoint_path}")
    return policy


def print_comparison(
    baseline_result: EvaluationResult,
    experiment_result: EvaluationResult,
):
    """Print comparison table."""
    print("\n")
    print("=" * 90)
    print("POLICY COMPARISON")
    print("=" * 90)
    print(f"{'Metric':<30} {'Baseline (Raw PPO)':<25} {'LLM-Init + PPO':<25}")
    print("-" * 90)
    print(f"{'Number of episodes':<30} {baseline_result.num_episodes:<25} {experiment_result.num_episodes:<25}")
    print(f"{'Success rate':<30} {baseline_result.success_rate:<25.2%} {experiment_result.success_rate:<25.2%}")
    print(f"{'Collision rate':<30} {baseline_result.collision_rate:<25.2%} {experiment_result.collision_rate:<25.2%}")
    print(f"{'Avg episode length':<30} {baseline_result.avg_episode_length:<25.1f} {experiment_result.avg_episode_length:<25.1f}")
    print(f"{'Avg episode reward':<30} {baseline_result.avg_episode_reward:<25.2f} {experiment_result.avg_episode_reward:<25.2f}")
    print(f"{'Std reward':<30} {baseline_result.std_episode_reward:<25.2f} {experiment_result.std_episode_reward:<25.2f}")
    print(f"{'Avg distance traveled':<30} {baseline_result.avg_distance_traveled:<25.2f} {experiment_result.avg_distance_traveled:<25.2f}")
    print("=" * 90)

    # Print summary
    print("\nSummary:")
    success_diff = experiment_result.success_rate - baseline_result.success_rate
    if success_diff > 0:
        print(f"  ✓ LLM-Init + PPO improves success rate by {success_diff:.2%}")
    elif success_diff < 0:
        print(f"  ✗ LLM-Init + PPO reduces success rate by {abs(success_diff):.2%}")
    else:
        print(f"  - No change in success rate")

    reward_diff = experiment_result.avg_episode_reward - baseline_result.avg_episode_reward
    if reward_diff > 0:
        print(f"  ✓ LLM-Init + PPO improves average reward by {reward_diff:.2f}")
    elif reward_diff < 0:
        print(f"  ✗ LLM-Init + PPO reduces average reward by {abs(reward_diff):.2f}")
    else:
        print(f"  - No change in average reward")

    length_diff = baseline_result.avg_episode_length - experiment_result.avg_episode_length
    if length_diff > 0:
        print(f"  ✓ LLM-Init + PPO is faster (shorter episodes by {length_diff:.1f} steps)")
    elif length_diff < 0:
        print(f"  ✗ LLM-Init + PPO is slower (longer episodes by {abs(length_diff):.1f} steps)")
    else:
        print(f"  - No change in episode length")
    print("=" * 90)
    print("\n")


def main():
    parser = argparse.ArgumentParser(description='Compare baseline PPO vs LLM-Init + PPO')
    parser.add_argument('--baseline', type=str, required=True,
                        help='Path to baseline PPO final model checkpoint (.pt)')
    parser.add_argument('--experiment', type=str, required=True,
                        help='Path to LLM-Init + PPO final model checkpoint (.pt)')
    parser.add_argument('--num-episodes', type=int, default=100,
                        help='Number of evaluation episodes')
    parser.add_argument('--world-size', type=float, default=10.0,
                        help='World size in meters')
    parser.add_argument('--num-static-obstacles', type=int, default=5,
                        help='Number of static obstacles')
    parser.add_argument('--num-dynamic-obstacles', type=int, default=0,
                        help='Number of dynamic obstacles')
    parser.add_argument('--max-episode-steps', type=int, default=200,
                        help='Maximum steps per episode')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for JSON results (optional)')

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load policies
    print("\nLoading policies...")
    baseline_policy = load_policy(args.baseline, device)
    experiment_policy = load_policy(args.experiment, device)

    # Evaluate baseline
    print("\nEvaluating baseline (raw PPO):")
    baseline_result = evaluate_policy(
        baseline_policy,
        args.num_episodes,
        args.world_size,
        args.num_static_obstacles,
        args.num_dynamic_obstacles,
        args.max_episode_steps,
        device,
        seed_offset=args.seed,
    )

    # Evaluate experiment
    print("\nEvaluating experiment (LLM-Init + PPO):")
    experiment_result = evaluate_policy(
        experiment_policy,
        args.num_episodes,
        args.world_size,
        args.num_static_obstacles,
        args.num_dynamic_obstacles,
        args.max_episode_steps,
        device,
        seed_offset=args.seed,
    )

    # Print comparison
    print_comparison(baseline_result, experiment_result)

    # Save results if output specified
    if args.output:
        result_dict = {
            'config': {
                'baseline_path': args.baseline,
                'experiment_path': args.experiment,
                'num_episodes': args.num_episodes,
                'world_size': args.world_size,
                'num_static_obstacles': args.num_static_obstacles,
                'num_dynamic_obstacles': args.num_dynamic_obstacles,
                'max_episode_steps': args.max_episode_steps,
                'seed': args.seed,
            },
            'baseline': {
                'success_rate': baseline_result.success_rate,
                'collision_rate': baseline_result.collision_rate,
                'avg_episode_length': baseline_result.avg_episode_length,
                'avg_episode_reward': baseline_result.avg_episode_reward,
                'std_episode_reward': baseline_result.std_episode_reward,
                'avg_distance_traveled': baseline_result.avg_distance_traveled,
            },
            'experiment': {
                'success_rate': experiment_result.success_rate,
                'collision_rate': experiment_result.collision_rate,
                'avg_episode_length': experiment_result.avg_episode_length,
                'avg_episode_reward': experiment_result.avg_episode_reward,
                'std_episode_reward': experiment_result.std_episode_reward,
                'avg_distance_traveled': experiment_result.avg_distance_traveled,
            },
            'difference': {
                'success_rate': experiment_result.success_rate - baseline_result.success_rate,
                'collision_rate': experiment_result.collision_rate - baseline_result.collision_rate,
                'avg_episode_length': experiment_result.avg_episode_length - baseline_result.avg_episode_length,
                'avg_episode_reward': experiment_result.avg_episode_reward - baseline_result.avg_episode_reward,
            },
        }

        with open(args.output, 'w') as f:
            json.dump(result_dict, f, indent=2)
        print(f"Saved results to: {args.output}")


if __name__ == '__main__':
    main()
