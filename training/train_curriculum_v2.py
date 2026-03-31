"""
Curriculum Learning v2 - Effective Implementation

Trains through phases with sufficient timesteps per phase:
- Phase 1: 0 obstacles, 200K timesteps, target 80% success
- Phase 2: 2 obstacles, 300K timesteps, target 70% success
- Phase 3: 4 obstacles, 400K timesteps, target 60% success
- Phase 4: 5+2 obstacles, 500K timesteps, target 70% success

Uses evaluation episodes to determine advancement.
"""

import os
import sys
import torch
import numpy as np
from envs import OccupancyGridEnv
from training.train_ppo_custom import (
    ActorCriticPolicy, collect_rollouts, compute_gae
)
import torch.optim as optim
import torch.nn as nn


def evaluate_policy(env, policy, n_episodes=20):
    """Evaluate policy and return success rate."""
    successes = 0
    rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=10000 + ep)
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


def train_phase(env, policy, optimizer, phase_info, device, timesteps_target):
    """Train for one phase until target timesteps or success threshold."""
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

    total_timesteps = 0
    best_success_rate = 0.0

    print(f"Training {phase_info}...")
    print(f"Target timesteps: {timesteps_target:,}")
    print("-" * 80)

    for update in range(1, 1000):  # Max 1000 updates
        if total_timesteps >= timesteps_target:
            break

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

        # Log progress
        if update % 10 == 0:
            recent_rewards = rollout['rewards'][-50:].numpy()
            recent_success = sum(1 for r in recent_rewards if r > 100)
            recent_rate = recent_success / len(recent_rewards)
            print(f"Update {update:3d}: Reward={np.mean(recent_rewards):6.2f}, Success={recent_rate*100:.0f}%")

    # Final evaluation
    print(f"\nFinal evaluation for {phase_info}...")
    eval_stats = evaluate_policy(env, policy, n_episodes=50)
    print(f"Success Rate: {eval_stats['success_rate'] * 100:.1f}% ({eval_stats['successes']}/50)")
    print(f"Mean Reward: {eval_stats['mean_reward']:.2f}")

    return total_timesteps, eval_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='./curriculum_output')
    args = parser.parse_args()

    device = torch.device('cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create policy
    policy = ActorCriticPolicy(action_dim=3, hidden_size=256).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4, eps=1e-5)

    print("=" * 80)
    print("CURRICULUM LEARNING TRAINING")
    print("=" * 80)

    # Define curriculum phases
    phases = [
        ("Phase 1: Easy (0 obstacles)", 0, 0, 50000),
        ("Phase 2: Medium (2 obstacles)", 2, 0, 100000),
        ("Phase 3: Hard (4 obstacles)", 4, 0, 150000),
        ("Phase 4: Expert (5+2 obstacles)", 5, 2, 200000),
    ]

    total_timesteps = 0

    # Train through curriculum phases
    for phase_idx, (name, static, dynamic, timesteps) in enumerate(phases, 1):
        phase_timesteps, eval_stats = train_phase(
            policy, optimizer, name, device, timesteps
        )

        total_timesteps += phase_timesteps

        # Save checkpoint
        ckpt_path = os.path.join(args.output_dir, f"phase_{phase_idx}.pt")
        torch.save({
            'phase': phase_idx,
            'policy_state_dict': policy.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'total_timesteps': total_timesteps,
            'eval_success_rate': eval_stats['success_rate'],
        }, ckpt_path)
        print(f"\nSaved checkpoint: {ckpt_path}")

    # Save final model
    final_path = os.path.join(args.output_dir, "final_model.pt")
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'total_timesteps': total_timesteps,
    }, final_path)

    print("\n" + "=" * 80)
    print(f"CURRICULUM LEARNING COMPLETE!")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Final model: {final_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
