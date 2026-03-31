"""
Visualize episodes from the 10M step trained model.
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs import OccupancyGridEnv
from training.train_ppo_custom import ActorCriticPolicy

def load_policy(checkpoint_path, device='cpu'):
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    policy = ActorCriticPolicy(action_dim=3, hidden_size=256).to(device)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()
    return policy

def run_episode(env, policy, seed=None, deterministic=True, device='cpu'):
    obs, info = env.reset(seed=seed)
    positions = []
    rewards = []
    done = False
    steps = 0
    total_reward = 0
    while not done and steps < env.max_episode_steps:
        positions.append(env.robot_state[0:2].copy())
        obs_tensor = {k: torch.FloatTensor(v).unsqueeze(0).to(device) for k, v in obs.items()}
        with torch.no_grad():
            if deterministic:
                mean, _, _ = policy(obs_tensor)
                action = mean.cpu().numpy()[0]
            else:
                action, _, _ = policy.get_action(obs_tensor, deterministic=False)
                action = action.cpu().numpy()[0]
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        total_reward += reward
        done = terminated or truncated
        steps += 1
    return {
        'positions': np.array(positions),
        'rewards': np.array(rewards),
        'total_reward': total_reward,
        'steps': steps,
        'goal_reached': info.get('goal_reached', False),
        'goal_position': env.grid_world.goal_position.copy(),
        'occupancy_grid': env.grid_world.occupancy_grid.copy()
    }

def plot_episode(result, episode_num, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Episode {episode_num} - {"✓ Success" if result["goal_reached"] else "✗ Failure"} | '
                f'Reward: {result["total_reward"]:.2f} | Steps: {result["steps"]}',
                fontsize=14, fontweight='bold')
    # Trajectory plot
    ax1 = axes[0]
    grid = result['occupancy_grid']
    ax1.imshow(grid.T, origin='lower', extent=[0, 10, 0, 10],
               cmap='gray', alpha=0.3)
    positions = result['positions']
    if len(positions) > 0:
        ax1.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, alpha=0.7)
        ax1.scatter(positions[0, 0], positions[0, 1], c='green', s=150,
                   marker='o', label='Start', edgecolors='black', zorder=5)
        ax1.scatter(positions[-1, 0], positions[-1, 1], c='red', s=150,
                   marker='X', label='End', edgecolors='black', zorder=5)
    goal = result['goal_position']
    ax1.scatter(goal[0], goal[1], c='gold', s=300, marker='*',
               label='Goal', edgecolors='black', linewidths=2, zorder=5)
    ax1.set_xlabel('X Position (m)', fontsize=11)
    ax1.set_ylabel('Y Position (m)', fontsize=11)
    ax1.set_title('Robot Trajectory (10M Step Model)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    # Reward over time
    ax2 = axes[1]
    rewards = result['rewards']
    ax2.plot(rewards, linewidth=1.5, color='blue', alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Step', fontsize=11)
    ax2.set_ylabel('Reward', fontsize=11)
    ax2.set_title('Reward per Step', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig

def main():
    checkpoint_path = './output/ppo_training_output_10m/checkpoint_10004480.pt'
    output_dir = './visualizations_10m'
    device = 'cpu'
    n_episodes = 5

    os.makedirs(output_dir, exist_ok=True)

    print('=' * 80)
    print('VISUALIZING 10M STEP PPO POLICY')
    print('=' * 80)
    print(f'Checkpoint: {checkpoint_path}')
    print(f'Output directory: {output_dir}')
    print(f'Episodes to visualize: {n_episodes}')
    print()

    policy = load_policy(checkpoint_path, device)
    print('Policy loaded successfully!')
    print()

    env = OccupancyGridEnv(
        world_width=10.0,
        world_height=10.0,
        num_static_obstacles=5,
        num_dynamic_obstacles=2,
        max_episode_steps=500,
        random_seed=42,
    )

    print('Running episodes...')
    print()

    successes = 0
    for i in range(n_episodes):
        print(f'Episode {i+1}/{n_episodes}...')
        result = run_episode(env, policy, seed=i+4000, deterministic=True, device=device)
        if result['goal_reached']:
            successes += 1
        status = "✓ SUCCESS" if result['goal_reached'] else "✗ Failure"
        print(f'  Status: {status}')
        print(f'  Reward: {result["total_reward"]:.2f}')
        print(f'  Steps: {result["steps"]}')
        print()
        save_path = os.path.join(output_dir, f'episode_{i+1:02d}_{"success" if result["goal_reached"] else "failure"}.png')
        plot_episode(result, i+1, save_path)
        plt.close()

    print('=' * 80)
    print('VISUALIZATION COMPLETE')
    print('=' * 80)
    print(f'Success rate in this sample: {successes/n_episodes*100:.0f}% ({successes}/{n_episodes})')
    print(f'All visualizations saved to: {output_dir}/')
    print()

if __name__ == '__main__':
    main()
