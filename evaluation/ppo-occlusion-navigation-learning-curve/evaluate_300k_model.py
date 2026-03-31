"""
Evaluate 300k step model success rate over 100 episodes.
"""
import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs import OccupancyGridEnv
from training.train_ppo_custom import ActorCriticPolicy

def load_policy(checkpoint_path, device='cpu'):
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    policy = ActorCriticPolicy(action_dim=3, hidden_size=256).to(device)
    if 'policy_state_dict' in checkpoint:
        policy.load_state_dict(checkpoint['policy_state_dict'])
    else:
        policy.load_state_dict(checkpoint)
    policy.eval()
    return policy

def run_episode(env, policy, seed=None, deterministic=True, device='cpu'):
    obs, info = env.reset(seed=seed)
    done = False
    steps = 0
    total_reward = 0.0
    while not done and steps < env.max_episode_steps:
        obs_tensor = {k: torch.FloatTensor(v).unsqueeze(0).to(device) for k, v in obs.items()}
        with torch.no_grad():
            if deterministic:
                mean, _, _ = policy(obs_tensor)
                action = mean.cpu().numpy()[0]
            else:
                action, _, _ = policy.get_action(obs_tensor, deterministic=False)
                action = action.cpu().numpy()[0]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        steps += 1
    return {
        'success': info.get('goal_reached', False),
        'reward': total_reward,
        'steps': steps
    }

def main():
    checkpoint_path = './output/ppo_training_output/checkpoint_301056.pt'
    device = 'cpu'
    n_episodes = 100

    print('=' * 80)
    print('EVALUATING 300K STEP MODEL')
    print('=' * 80)
    print(f'Checkpoint: {checkpoint_path}')
    print(f'Episodes: {n_episodes}')
    print()

    policy = load_policy(checkpoint_path, device)
    print('Policy loaded')

    env = OccupancyGridEnv(
        world_width=10.0,
        world_height=10.0,
        num_static_obstacles=5,
        num_dynamic_obstacles=2,
        max_episode_steps=500,
    )
    print('Environment created')
    print()

    successes = 0
    total_reward_sum = 0
    total_steps_sum = 0
    success_rewards = []
    failure_rewards = []

    for i in range(n_episodes):
        result = run_episode(env, policy, seed=i+42, deterministic=True, device=device)
        if result['success']:
            successes += 1
            success_rewards.append(result['reward'])
        else:
            failure_rewards.append(result['reward'])
        total_reward_sum += result['reward']
        total_steps_sum += result['steps']

        if (i + 1) % 10 == 0:
            print(f'  Completed {i+1}/{n_episodes} - Success rate: {successes/(i+1)*100:.1f}%')

    print()
    print('=' * 80)
    print('FINAL RESULTS (300K Step Model)')
    print('=' * 80)
    print(f'Total timesteps: 301,056')
    print(f'Total episodes: {n_episodes}')
    print(f'Successful: {successes}')
    print(f'Failed: {n_episodes - successes}')
    print(f'Success rate: {successes / n_episodes * 100:.2f}%')
    print(f'Mean reward: {total_reward_sum / n_episodes:.2f}')
    print(f'Mean steps per episode: {total_steps_sum / n_episodes:.1f}')
    if successes > 0:
        print(f'Mean reward (successful): {np.mean(success_rewards):.2f}')
    print()

    # Save results
    os.makedirs('./output/evaluation_results', exist_ok=True)
    with open('./output/evaluation_results/300k_model_results.txt', 'w') as f:
        f.write(f'Total timesteps: 301,056\n')
        f.write(f'Episodes evaluated: {n_episodes}\n')
        f.write(f'Success rate: {successes / n_episodes * 100:.2f}%\n')
        f.write(f'Mean reward: {total_reward_sum / n_episodes:.2f}\n')

if __name__ == '__main__':
    main()
