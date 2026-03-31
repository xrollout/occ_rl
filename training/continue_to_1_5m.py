"""
Continue training PPO model from 500K to 1.5M timesteps


This script loads the best 500K checkpoint and continues training
for an additional 1M timesteps to reach 1.5M total.
"""
import os
import sys
import torch
import numpy as np


sys.path.insert(0, '/Users/bobinding/Documents/robot/xrollout')

from envs import OccupancyGridEnv
from training.train_ppo_custom import (
    ActorCriticPolicy, collect_rollouts, compute_gae
)
import torch.optim as optim
import torch.nn as nn

def evaluate_policy(env, policy, n_episodes=20):
    """Quick evaluation."""
    successes = 0
    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep + 10000)
        done = False
        steps = 0
        ep_reward = 0
        while not done and steps < 500:
            obs_tensor = {k: torch.FloatTensor(v).unsqueeze(0) for k, v in obs.items()}
            with torch.no_grad():
                action, _, _ = policy.get_action(obs_tensor, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action[0])
            ep_reward += reward
            done = terminated or truncated
            steps += 1
        rewards.append(ep_reward)
        if info.get('goal_reached', False):
            successes += 1
    return {
        'success_rate': successes / n_episodes,
        'mean_reward': np.mean(rewards),
        'successes': successes,
    }

def main():
    # Configuration
    checkpoint_path = 'ppo_training_output_500k/final_model.pt'
    output_dir = './ppo_1_5m_output'
    additional_timesteps = 1000000  # 1M additional timesteps
    device = torch.device('cpu')

    os.makedirs(output_dir, exist_ok=True)

    print('=' * 80)
    print('CONTINUING PPO TRAINING: 500K -> 1.5M TIMESTEPS')
    print('=' * 80)
    print()
    print(f'Checkpoint: {checkpoint_path}')
    print(f'Additional timesteps: {additional_timesteps:,}')
    print(f'Target total: 1,500,000 timesteps')
    print()

    # Load checkpoint
    print('Loading checkpoint...')
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    # Create policy and load state
    policy = ActorCriticPolicy(action_dim=3, hidden_size=256).to(device)
    policy.load_state_dict(checkpoint['policy_state_dict'])

    # Create optimizer and load state
    optimizer = optim.Adam(policy.parameters(), lr=3e-4, eps=1e-5)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    previous_timesteps = checkpoint.get('total_timesteps', 500000)
    target_timesteps = previous_timesteps + additional_timesteps

    print(f'Previous timesteps: {previous_timesteps:,}')
    print(f'Target timesteps: {target_timesteps:,}')
    print('Loaded successfully!')
    print()

    # Create environment
    env = OccupancyGridEnv(
        world_width=10.0,
        world_height=10.0,
        num_static_obstacles=5,
        num_dynamic_obstacles=2,
        max_episode_steps=500,
        random_seed=42,
    )

    # Training hyperparameters
    n_steps = 2048
    batch_size = 64
    n_epochs = 10
    gamma = 0.99
    gae_lambda = 0.95
    clip_range = 0.2
    vf_coef = 0.5
    ent_coef = 0.01
    max_grad_norm = 0.5

    print('=' * 80)
    print('STARTING TRAINING')
    print('=' * 80)
    print()

    total_timesteps = previous_timesteps
    update_num = 0
    best_success_rate = 0.0

    while total_timesteps < target_timesteps:
        # Collect rollouts
        rollout = collect_rollouts(env, policy, n_steps, device)

        # Compute GAE
        advantages, returns = compute_gae(
            rollout['rewards'], rollout['values'], rollout['dones'],
            gamma, gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Update policy
        n_samples = len(rollout['rewards'])
        indices = np.arange(n_samples)

        for epoch in range(n_epochs):
            np.random.shuffle(indices)
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                mb_idx = indices[start:end]

                mb_obs = {k: v[mb_idx].to(device) for k, v in rollout['obs'].items()}
                mb_actions = rollout['actions'][mb_idx].to(device)
                mb_old_log = rollout['log_probs'][mb_idx].to(device)
                mb_adv = advantages[mb_idx].to(device)
                mb_ret = returns[mb_idx].to(device)

                mean, std, values = policy(mb_obs)
                dist = torch.distributions.Normal(mean, std)
                log_probs = dist.log_prob(mb_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = torch.exp(log_probs - mb_old_log)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * (values.squeeze(-1) - mb_ret).pow(2).mean()

                loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()

        total_timesteps += n_samples
        update_num += 1

        # Log progress
        if update_num % 10 == 0:
            recent_rewards = rollout['rewards'][-50:].numpy()
            recent_success = sum(1 for r in recent_rewards if r > 100)
            print(f"Update {update_num:3d}: Reward={np.mean(recent_rewards):6.2f}, Success={recent_success}/50")

    print()
    print('=' * 80)
    print('CONTINUED TRAINING COMPLETE')
    print('=' * 80)
    print(f'Total timesteps: {total_timesteps:,}')
    print(f'Additional timesteps: {total_timesteps - previous_timesteps:,}')

    # Save checkpoint
    checkpoint_path = os.path.join(output_dir, f'checkpoint_{total_timesteps}.pt')
    torch.save({
        'total_timesteps': total_timesteps,
        'policy_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f'Saved: {checkpoint_path}')

    # Final evaluation
    print('\nFinal evaluation:')
    eval_stats = evaluate_policy(env, policy, n_episodes=50)
    print(f"Success Rate: {eval_stats['success_rate']*100:.1f}% ({eval_stats['successes']}/50)")
    print(f"Mean Reward: {eval_stats['mean_reward']:.2f}")

if __name__ == '__main__':
    main()