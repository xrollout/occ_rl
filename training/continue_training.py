"""
Continue training PPO model from checkpoint

This script loads a trained checkpoint and continues training
for additional timesteps.
"""

import os
import sys
import torch
import numpy as np
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs import OccupancyGridEnv
from training.train_ppo_custom import (
    ActorCriticPolicy, collect_rollouts, compute_gae
)
import torch.optim as optim
import torch.nn as nn

def continue_training(
    checkpoint_path: str,
    additional_timesteps: int,
    output_dir: str,
    device_str: str = 'cpu'
):
    """Continue training from checkpoint."""

    device = torch.device(device_str)
    os.makedirs(output_dir, exist_ok=True)

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    # Create policy and load state
    policy = ActorCriticPolicy(action_dim=3, hidden_size=256).to(device)
    policy.load_state_dict(checkpoint['policy_state_dict'])

    # Create optimizer and load state
    optimizer = optim.Adam(policy.parameters(), lr=3e-4, eps=1e-5)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    previous_timesteps = checkpoint.get('total_timesteps', 0)
    target_timesteps = previous_timesteps + additional_timesteps

    print(f"Previous timesteps: {previous_timesteps:,}")
    print(f"Target timesteps: {target_timesteps:,}")
    print(f"Additional timesteps to train: {additional_timesteps:,}")
    print()

    # Training hyperparameters
    n_steps = 2048
    batch_size = 64
    n_epochs = 10
    lr = 3e-4
    gamma = 0.99
    gae_lambda = 0.95
    clip_range = 0.2
    vf_coef = 0.5
    ent_coef = 0.01
    max_grad_norm = 0.5

    # Create environment (full difficulty)
    env = OccupancyGridEnv(
        world_width=10.0,
        world_height=10.0,
        num_static_obstacles=5,
        num_dynamic_obstacles=2,
        max_episode_steps=500,
        random_seed=42,
    )

    print("=" * 80)
    print(f"CONTINUING TRAINING FOR {additional_timesteps:,} TIMESTEPS")
    print("=" * 80)
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
    print("=" * 80)
    print("CONTINUED TRAINING COMPLETE")
    print("=" * 80)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Additional timesteps: {total_timesteps - previous_timesteps:,}")

    # Save checkpoint
    checkpoint_path = os.path.join(output_dir, f'checkpoint_{total_timesteps}.pt')
    torch.save({
        'total_timesteps': total_timesteps,
        'policy_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Saved: {checkpoint_path}")

    # Final evaluation
    print("\nFinal evaluation:")
    eval_stats = evaluate(env, policy, n_episodes=50)
    print(f"Success Rate: {eval_stats['success_rate']*100:.1f}% ({eval_stats['successes']}/50)")
    print(f"Mean Reward: {eval_stats['mean_reward']:.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Continue training from checkpoint')
    parser.add_argument('--checkpoint', type=str, default='./ppo_training_output/final_model.pt',
                      help='Path to checkpoint to continue from')
    parser.add_argument('--additional-timesteps', type=int, default=700000,
                      help='Additional timesteps to train (300k + 700k = 1M)')
    parser.add_argument('--output-dir', type=str, default='./ppo_training_output_1m',
                      help='Output directory for new checkpoints')
    parser.add_argument('--device', type=str, default='cpu',
                      help='Device to use for training')
    args = parser.parse_args()

    continue_training(
        checkpoint_path=args.checkpoint,
        additional_timesteps=args.additional_timesteps,
        output_dir=args.output_dir,
        device_str=args.device
    )
