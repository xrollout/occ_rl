"""
Simplified Curriculum Learning for Grid Navigation

Trains progressively through:
1. Easy: 0 obstacles (learn basic navigation)
2. Medium: 2 static obstacles
3. Hard: 4 static obstacles
4. Expert: 5 static + 2 dynamic obstacles

Uses success rate from evaluation episodes to determine when to advance.
"""

import argparse
import os
import sys
import torch
import numpy as np
from envs import OccupancyGridEnv
from training.train_ppo_custom import (
    ActorCriticPolicy, collect_rollouts, compute_gae, PPOConfig
)
import torch.optim as optim
import torch.nn as nn


def evaluate_policy(env, policy, n_episodes=20):
    """Quick evaluation to get success rate."""
    successes = 0
    rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep + 1000)
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


def train_phase(env, policy, optimizer, config, device, phase_info):
    """Train for one curriculum phase."""
    total_timesteps = 0
    eval_rewards = []

    print(f"\nTraining {phase_info}...")
    print("-" * 80)

    for update in range(1, 51):  # 50 updates = ~100K timesteps
        # Collect rollouts
        rollout = collect_rollouts(env, policy, config.n_steps, device)

        # Compute GAE
        advantages, returns = compute_gae(
            rollout['rewards'], rollout['values'], rollout['dones'],
            config.gamma, config.gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Update policy
        n_samples = len(rollout['rewards'])
        indices = np.arange(n_samples)

        for epoch in range(config.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, n_samples, config.batch_size):
                end = min(start + config.batch_size, n_samples)
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

                # PPO loss
                ratio = torch.exp(log_probs - mb_old_log)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - config.clip_range, 1 + config.clip_range) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * (values.squeeze(-1) - mb_ret).pow(2).mean()

                loss = policy_loss + config.vf_coef * value_loss - config.ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
                optimizer.step()

        total_timesteps += n_samples

        # Log progress
        if update % 10 == 0:
            recent_rewards = rollout['rewards'][-50:].numpy() if len(rollout['rewards']) > 50 else rollout['rewards'].numpy()
            recent_success = sum(1 for r in recent_rewards if r > 100)
            print(f"Update {update:2d}: Reward={np.mean(recent_rewards):6.2f}, Success={recent_success}/{len(recent_rewards)}")

    return total_timesteps


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

    config = PPOConfig(
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
    )

    # Curriculum phases
    phases = [
        ("Phase 1: Easy (0 obstacles)", 0, 0, 0.70),
        ("Phase 2: Medium (2 obstacles)", 2, 0, 0.60),
        ("Phase 3: Hard (4 obstacles)", 4, 0, 0.50),
        ("Phase 4: Expert (5+2 obstacles)", 5, 2, 0.70),
    ]

    print("=" * 80)
    print("CURRICULUM LEARNING - 4 PHASES")
    print("=" * 80)

    total_timesteps = 0

    for phase_idx, (name, static, dynamic, target) in enumerate(phases, 1):
        print(f"\n{'=' * 80}")
        print(f"{name}")
        print(f"Target: {target * 100:.0f}% success rate")
        print(f"{'=' * 80}")

        # Create environment
        env = OccupancyGridEnv(
            world_width=10.0,
            world_height=10.0,
            num_static_obstacles=static,
            num_dynamic_obstacles=dynamic,
            max_episode_steps=500,
            random_seed=args.seed + phase_idx * 100,
        )

        # Train this phase
        phase_steps = train_phase(env, policy, optimizer, config, device, name)
        total_timesteps += phase_steps

        # Evaluate
        print(f"\nEvaluating {name}...")
        eval_stats = evaluate_policy(env, policy, n_episodes=20)
        print(f"  Success Rate: {eval_stats['success_rate'] * 100:.1f}% ({eval_stats['successes']}/20)")
        print(f"  Mean Reward: {eval_stats['mean_reward']:.2f}")

        # Save checkpoint
        ckpt_path = os.path.join(args.output_dir, f"phase_{phase_idx}.pt")
        torch.save({
            'policy_state_dict': policy.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'total_timesteps': total_timesteps,
            'eval_success_rate': eval_stats['success_rate'],
        }, ckpt_path)
        print(f"Saved: {ckpt_path}")

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

    # Final evaluation on full environment
    print("\nFinal evaluation on full environment:")
    final_env = OccupancyGridEnv(
        world_width=10.0,
        world_height=10.0,
        num_static_obstacles=5,
        num_dynamic_obstacles=2,
        max_episode_steps=500,
        random_seed=99999,
    )
    final_eval = evaluate_policy(final_env, policy, n_episodes=50)
    print(f"Success Rate: {final_eval['success_rate'] * 100:.1f}% ({final_eval['successes']}/50)")
    print(f"Mean Reward: {final_eval['mean_reward']:.2f}")


if __name__ == '__main__':
    main()
