"""
Collect LLM Demonstrations

This script runs the environment with the LLM as the policy and collects
(observation, action, reward) transitions for behavior cloning.
"""

import argparse
import os
import sys
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs import OccupancyGridEnv
from llm_teacher.llm_teacher import LLMTeacher, LLMConfig


def collect_demonstrations(
    num_episodes: int,
    world_size: float,
    num_static_obstacles: int,
    num_dynamic_obstacles: int,
    max_episode_steps: int,
    seed: int,
    use_cache: bool,
    difficulty: str = "easy",
    chunk_size: int = 1,
) -> dict:
    """
    Collect demonstrations by running LLM policy in environment.

    Args:
        num_episodes: Number of episodes to collect.
        world_size: Size of world in meters.
        num_static_obstacles: Number of static obstacles.
        num_dynamic_obstacles: Number of dynamic obstacles.
        max_episode_steps: Maximum steps per episode.
        seed: Random seed.
        use_cache: Whether to use cached LLM responses.
        difficulty: "easy" (random obstacles) or "hard" (structured obstacles requiring detour).
        chunk_size: Number of actions per LLM query (1 = query every step).
            chunk_size > 1 reduces API calls by factor of chunk_size.

    Returns:
        Dataset dictionary with all collected data.
    """
    # Create environment
    hard_scenario = (difficulty == "hard")
    env = OccupancyGridEnv(
        world_width=world_size,
        world_height=world_size,
        num_static_obstacles=num_static_obstacles,
        num_dynamic_obstacles=num_dynamic_obstacles,
        max_episode_steps=max_episode_steps,
        random_seed=seed,
        hard_scenario=hard_scenario,
    )

    # Create LLM teacher
    llm_teacher = LLMTeacher()

    # Storage
    all_obs_grid = []
    all_obs_robot_pose = []
    all_obs_target_relative = []
    all_obs_velocity = []
    all_actions = []
    all_rewards = []
    all_dones = []
    episode_infos = []

    # Collection loop
    print(f"Collecting {num_episodes} episodes...")
    print(f"World size: {world_size}m, Obstacles: {num_static_obstacles} static + {num_dynamic_obstacles} dynamic")
    print(f"Difficulty: {difficulty}, Chunk size: {chunk_size}")
    print(f"Expected API calls: ~{int(num_episodes * max_episode_steps / chunk_size)} (vs {num_episodes * max_episode_steps} for chunk_size=1)")

    successes = 0
    collisions = 0
    total_steps = 0
    total_api_calls = 0

    for episode_idx in tqdm(range(num_episodes)):
        obs, info = env.reset()
        done = False
        episode_steps = 0
        episode_reward = 0.0
        success = False
        collision = False

        while not done and episode_steps < max_episode_steps:
            remaining_steps = max_episode_steps - episode_steps
            current_chunk_size = min(chunk_size, remaining_steps)

            if current_chunk_size > 1:
                # Get chunk of actions from LLM teacher
                total_api_calls += 1
                actions = llm_teacher.get_action_chunk(obs, current_chunk_size, use_cache=use_cache)

                # Execute each action in the chunk
                for action in actions:
                    if done:
                        break

                    # Store transition (store the observation BEFORE executing this action)
                    all_obs_grid.append(obs['occupancy_grid'].copy())
                    all_obs_robot_pose.append(obs['robot_pose'].copy())
                    all_obs_target_relative.append(obs['target_relative'].copy())
                    all_obs_velocity.append(obs['velocity'].copy())
                    all_actions.append(action.copy())

                    # Step environment
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                    all_rewards.append(reward)
                    all_dones.append(float(done))

                    episode_reward += reward
                    episode_steps += 1

                    if info.get('goal_reached', False):
                        success = True
                        successes += 1
                    if info.get('collision', False):
                        collision = True
                        collisions += 1
            else:
                # Single-step mode - query every step
                total_api_calls += 1
                action = llm_teacher.get_action(obs, use_cache=use_cache)

                # Store transition
                all_obs_grid.append(obs['occupancy_grid'].copy())
                all_obs_robot_pose.append(obs['robot_pose'].copy())
                all_obs_target_relative.append(obs['target_relative'].copy())
                all_obs_velocity.append(obs['velocity'].copy())
                all_actions.append(action.copy())

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                all_rewards.append(reward)
                all_dones.append(float(done))

                episode_reward += reward
                episode_steps += 1

                if info.get('goal_reached', False):
                    success = True
                    successes += 1
                if info.get('collision', False):
                    collision = True
                    collisions += 1

        total_steps += episode_steps

        episode_infos.append({
            'episode_idx': episode_idx,
            'episode_steps': episode_steps,
            'episode_reward': episode_reward,
            'success': success,
            'collision': collision,
        })

    # Convert to numpy arrays
    dataset = {
        'obs_grid': np.array(all_obs_grid),
        'obs_robot_pose': np.array(all_obs_robot_pose),
        'obs_target_relative': np.array(all_obs_target_relative),
        'obs_velocity': np.array(all_obs_velocity),
        'actions': np.array(all_actions),
        'rewards': np.array(all_rewards),
        'dones': np.array(all_dones),
        'episode_infos': episode_infos,
        'metadata': {
            'num_episodes': num_episodes,
            'world_size': world_size,
            'num_static_obstacles': num_static_obstacles,
            'num_dynamic_obstacles': num_dynamic_obstacles,
            'max_episode_steps': max_episode_steps,
            'seed': seed,
            'difficulty': difficulty,
            'chunk_size': chunk_size,
            'total_transitions': total_steps,
            'total_api_calls': total_api_calls,
            'success_rate': successes / num_episodes,
            'collision_rate': collisions / num_episodes,
            'avg_episode_length': total_steps / num_episodes,
        }
    }

    # Print statistics
    print()
    print("=" * 60)
    print("COLLECTION STATISTICS")
    print("=" * 60)
    print(f"Total episodes: {num_episodes}")
    print(f"Total transitions: {total_steps}")
    print(f"Success rate: {successes / num_episodes:.2%}")
    print(f"Collision rate: {collisions / num_episodes:.2%}")
    print(f"Average episode length: {total_steps / num_episodes:.1f} steps")
    print("=" * 60)

    return dataset


def main():
    parser = argparse.ArgumentParser(description='Collect LLM demonstration dataset')
    parser.add_argument('--num-episodes', type=int, default=200,
                        help='Number of episodes to collect')
    parser.add_argument('--world-size', type=float, default=10.0,
                        help='World size in meters')
    parser.add_argument('--num-static-obstacles', type=int, default=5,
                        help='Number of static obstacles')
    parser.add_argument('--num-dynamic-obstacles', type=int, default=0,
                        help='Number of dynamic obstacles')
    parser.add_argument('--max-episode-steps', type=int, default=200,
                        help='Maximum steps per episode')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--difficulty', type=str, default='easy', choices=['easy', 'hard'],
                        help='Difficulty: easy = random obstacles, hard = structured obstacles requiring detour')
    parser.add_argument('--chunk-size', type=int, default=1,
                        help='Number of actions per LLM query (>=1). Chunking reduces API calls.')
    parser.add_argument('--output', type=str,
                        default='./data/llm_demonstrations/dataset_{seed}.npz',
                        help='Output path for dataset (.npz format)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable LLM response caching')

    args = parser.parse_args()

    # Create output directory
    output_path = args.output.format(seed=args.seed)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Collect dataset
    dataset = collect_demonstrations(
        num_episodes=args.num_episodes,
        world_size=args.world_size,
        num_static_obstacles=args.num_static_obstacles,
        num_dynamic_obstacles=args.num_dynamic_obstacles,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
        use_cache=not args.no_cache,
        difficulty=args.difficulty,
        chunk_size=args.chunk_size,
    )

    # Save dataset
    np.savez_compressed(output_path, **dataset)
    print(f"\nSaved dataset to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")


if __name__ == '__main__':
    main()
