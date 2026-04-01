"""
Final Curriculum Learning Implementation

4-phase curriculum to reach 60-80% success rate:
- Phase 1: 0 obstacles, 200K steps, target 80%
- Phase 2: 2 obstacles, 200K steps, target 70%
- Phase 3: 4 obstacles, 200K steps, target 60%
- Phase 4: 5+2 obstacles, 400K steps, target 70%
"""

import os, sys, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs import OccupancyGridEnv
from training.train_ppo_custom import (
    ActorCriticPolicy, collect_rollouts, compute_gae
)
import torch.optim as optim
import torch.nn as nn

def evaluate(env, policy, n_episodes=20):
    """Quick evaluation."""
    successes = 0
    total_reward = 0.0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep + 10000)
        done = False
        steps = 0
        episode_reward = 0.0
        while not done and steps < 500:
            obs_tensor = {k: torch.FloatTensor(v).unsqueeze(0) for k, v in obs.items()}
            with torch.no_grad():
                action, _, _ = policy.get_action(obs_tensor, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action[0])
            done = terminated or truncated
            episode_reward += reward
            steps += 1
        total_reward += episode_reward
        if info.get('goal_reached', False):
            successes += 1
    return {
        'success_rate': successes / n_episodes,
        'successes': successes,
        'mean_reward': total_reward / n_episodes
    }

def train_phase(env, policy, optimizer, n_steps, batch_size, n_epochs, device, total_timesteps_target):
    """Train for one phase."""
    gamma, gae_lambda = 0.99, 0.95
    clip_range, vf_coef, ent_coef = 0.2, 0.5, 0.01
    max_grad_norm = 0.5

    total_timesteps = 0
    update_num = 0

    while total_timesteps < total_timesteps_target:
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
            recent_rewards = rollout['rewards'][-50:].numpy() if len(rollout['rewards']) > 50 else rollout['rewards'].numpy()
            recent_success = sum(1 for r in recent_rewards if r > 100)
            print(f"Update {update_num:3d}: Reward={np.mean(recent_rewards):6.2f}, Success={recent_success}/{len(recent_rewards)}")

    return total_timesteps


def main():
    # Curriculum phases: (name, static, dynamic, target_timesteps)
    phases = [
        ("Phase 1: Easy (0 obstacles)", 0, 0, 100000),
        ("Phase 2: Medium (2 obstacles)", 2, 0, 150000),
        ("Phase 3: Hard (4 obstacles)", 4, 0, 200000),
        ("Phase 4: Expert (5+2 obstacles)", 5, 2, 250000),
    ]

    device = torch.device('cpu')
    output_dir = './curriculum_v2_output'
    os.makedirs(output_dir, exist_ok=True)

    # Create policy
    policy = ActorCriticPolicy(action_dim=3, hidden_size=256).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4, eps=1e-5)

    print("=" * 80)
    print("CURRICULUM LEARNING v2")
    print("=" * 80)

    total_timesteps = 0
    overall_best_success = 0.0

    # Train through phases
    for phase_idx, (name, static, dynamic, target_timesteps) in enumerate(phases, 1):
        print(f"\n{'='*80}")
        print(f"{name}")
        print(f"Target timesteps: {target_timesteps:,}")
        print(f"{'='*80}")

        # Create environment for this phase
        env = OccupancyGridEnv(
            world_width=10.0,
            world_height=10.0,
            num_static_obstacles=static,
            num_dynamic_obstacles=dynamic,
            max_episode_steps=500,
            random_seed=42 + phase_idx * 100,
        )

        # Train this phase
        phase_timesteps = train_phase(
            env, policy, optimizer,
            n_steps=2048, batch_size=64, n_epochs=10,
            device=device, total_timesteps_target=target_timesteps
        )
        total_timesteps += phase_timesteps

        # Evaluate phase
        print(f"\nEvaluating {name}...")
        eval_stats = evaluate(env, policy, n_episodes=50)
        print(f"Success Rate: {eval_stats['success_rate']*100:.1f}% ({eval_stats['successes']}/50)")
        print(f"Mean Reward: {eval_stats['mean_reward']:.2f}")

        if eval_stats['success_rate'] > overall_best_success:
            overall_best_success = eval_stats['success_rate']

        # Save checkpoint
        ckpt_path = os.path.join(output_dir, f"phase_{phase_idx}.pt")
        torch.save({
            'phase': phase_idx,
            'policy_state_dict': policy.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'total_timesteps': total_timesteps,
            'eval_success_rate': eval_stats['success_rate'],
        }, ckpt_path)
        print(f"Saved: {ckpt_path}")

    # Save final model
    final_path = os.path.join(output_dir, "final_model.pt")
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'total_timesteps': total_timesteps,
    }, final_path)

    print("\n" + "=" * 80)
    print("CURRICULUM LEARNING COMPLETE!")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Best success rate: {overall_best_success*100:.1f}%")
    print(f"Final model: {final_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
