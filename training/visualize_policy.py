"""
Visualize the trained PPO policy on the occupancy grid environment.

This script loads a trained checkpoint and generates visualizations including:
- Trajectory plots showing robot paths
- Grid occupancy with obstacle locations
- Success/failure analysis
- Optional: Animated GIF of episodes
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import imageio

sys.path.insert(0, '/Users/bobinding/Documents/robot/xrollout')

from envs import OccupancyGridEnv
from training.train_ppo_custom import ActorCriticPolicy

def load_policy(checkpoint_path, device='cpu'):
    """Load trained policy from checkpoint."""
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    policy = ActorCriticPolicy(action_dim=3, hidden_size=256).to(device)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()
    return policy, checkpoint.get('total_timesteps', 0)

def run_episode(env, policy, seed=None, deterministic=True, device='cpu'):
    """Run a single episode and record trajectory."""
    obs, info = env.reset(seed=seed)

    trajectory = {
        'positions': [],
        'actions': [],
        'rewards': [],
        'dones': [],
        'goal_reached': False,
        'total_reward': 0,
        'steps': 0
    }

    done = False
    steps = 0

    while not done and steps < env.max_episode_steps:
        # Record current position
        robot_pos = env.robot_position.copy()
        trajectory['positions'].append(robot_pos)

        # Get action from policy
        obs_tensor = {k: torch.FloatTensor(v).unsqueeze(0).to(device) for k, v in obs.items()}
        with torch.no_grad():
            if deterministic:
                mean, _, _ = policy(obs_tensor)
                action = mean.cpu().numpy()[0]
            else:
                action, _, _ = policy.get_action(obs_tensor, deterministic=False)
                action = action.cpu().numpy()[0]

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        trajectory['actions'].append(action)
        trajectory['rewards'].append(reward)
        trajectory['total_reward'] += reward

        done = terminated or truncated
        steps += 1

        if done:
            trajectory['goal_reached'] = info.get('goal_reached', False)

    trajectory['steps'] = steps
    trajectory['positions'] = np.array(trajectory['positions'])
    trajectory['actions'] = np.array(trajectory['actions'])
    trajectory['rewards'] = np.array(trajectory['rewards'])

    return trajectory

def visualize_episode(env, trajectory, episode_num=0, save_path=None):
    """Create a static visualization of an episode."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'Episode {episode_num} - {"Success" if trajectory["goal_reached"] else "Failure"} | '
                f'Reward: {trajectory["total_reward"]:.2f} | Steps: {trajectory["steps"]}',
                fontsize=14, fontweight='bold')

    # Main trajectory plot
    ax1 = axes[0, 0]
    occupancy_grid = env.occupancy_grid
    grid_extent = [0, env.world_width, 0, env.world_height]
    ax1.imshow(occupancy_grid.T, origin='lower', extent=grid_extent,
               cmap='gray', alpha=0.3)

    # Plot trajectory
    positions = trajectory['positions']
    if len(positions) > 0:
        # Color gradient based on time
        colors = plt.cm.viridis(np.linspace(0, 1, len(positions)))
        for i in range(len(positions) - 1):
            ax1.plot([positions[i, 0], positions[i+1, 0]],
                    [positions[i, 1], positions[i+1, 1]],
                    color=colors[i], linewidth=2, alpha=0.7)

        # Start and end markers
        ax1.scatter(positions[0, 0], positions[0, 1], c='green', s=200,
                   marker='o', label='Start', edgecolors='black', zorder=5)
        ax1.scatter(positions[-1, 0], positions[-1, 1], c='red', s=200,
                   marker='X', label='End', edgecolors='black', zorder=5)

    # Goal position
    goal = env.goal_position
    ax1.scatter(goal[0], goal[1], c='gold', s=300, marker='*',
               label='Goal', edgecolors='black', linewidths=2, zorder=5)

    ax1.set_xlabel('X Position (m)', fontsize=11)
    ax1.set_ylabel('Y Position (m)', fontsize=11)
    ax1.set_title('Robot Trajectory', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, env.world_width)
    ax1.set_ylim(0, env.world_height)

    # Reward over time
    ax2 = axes[0, 1]
    rewards = trajectory['rewards']
    ax2.plot(rewards, linewidth=1.5, color='blue', alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Step', fontsize=11)
    ax2.set_ylabel('Reward', fontsize=11)
    ax2.set_title('Reward per Step', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Action visualization
    ax3 = axes[1, 0]
    actions = trajectory['actions']
    if len(actions) > 0:
        time_steps = np.arange(len(actions))
        ax3.plot(time_steps, actions[:, 0], label='Vx (m/s)', linewidth=1.5)
        ax3.plot(time_steps, actions[:, 1], label='Vy (m/s)', linewidth=1.5)
        ax3.plot(time_steps, np.degrees(actions[:, 2]), label='Omega (deg/s)', linewidth=1.5)
    ax3.set_xlabel('Step', fontsize=11)
    ax3.set_ylabel('Action Value', fontsize=11)
    ax3.set_title('Robot Actions Over Time', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Episode summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    summary_text = f"""
    Episode Summary

    Status: {'✓ SUCCESS' if trajectory['goal_reached'] else '✗ Failure'}

    Total Steps: {trajectory['steps']}
    Total Reward: {trajectory['total_reward']:.2f}

    Average Reward/Step: {trajectory['total_reward']/max(trajectory['steps'],1):.3f}

    Start Position: ({positions[0,0]:.2f}, {positions[0,1]:.2f})
    End Position: ({positions[-1,0]:.2f}, {positions[-1,1]:.2f})
    Goal Position: ({goal[0]:.2f}, {goal[1]:.2f})

    Distance to Goal: {np.linalg.norm(positions[-1] - goal):.2f}m
    """
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")

    return fig

def run_multiple_episodes(env, policy, n_episodes=10, deterministic=True, device='cpu'):
    """Run multiple episodes and collect statistics."""
    results = []

    for i in range(n_episodes):
        print(f"Running episode {i+1}/{n_episodes}...")
        trajectory = run_episode(env, policy, seed=i+1000, deterministic=deterministic, device=device)
        results.append(trajectory)

    return results

def plot_summary_statistics(results, save_path=None):
    """Plot summary statistics for multiple episodes."""
    n_episodes = len(results)
    successes = sum(1 for r in results if r['goal_reached'])
    total_rewards = [r['total_reward'] for r in results]
    steps = [r['steps'] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Batch Evaluation Results (n={n_episodes})\n'
                f'Success Rate: {successes}/{n_episodes} ({100*successes/n_episodes:.1f}%)',
                fontsize=14, fontweight='bold')

    # Success rate pie chart
    ax1 = axes[0, 0]
    labels = ['Success', 'Failure']
    sizes = [successes, n_episodes - successes]
    colors = ['#90EE90', '#FFB6C1']
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Success Rate', fontsize=12, fontweight='bold')

    # Total reward distribution
    ax2 = axes[0, 1]
    ax2.hist(total_rewards, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(total_rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(total_rewards):.1f}')
    ax2.set_xlabel('Total Reward', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Reward Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Steps distribution
    ax3 = axes[1, 0]
    ax3.hist(steps, bins=20, color='lightcoral', edgecolor='black', alpha=0.7)
    ax3.axvline(np.mean(steps), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(steps):.1f}')
    ax3.set_xlabel('Steps', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Episode Length Distribution', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Reward vs Steps scatter
    ax4 = axes[1, 1]
    colors = ['green' if r['goal_reached'] else 'red' for r in results]
    ax4.scatter(steps, total_rewards, c=colors, alpha=0.6, s=100, edgecolors='black')
    ax4.set_xlabel('Steps', fontsize=11)
    ax4.set_ylabel('Total Reward', fontsize=11)
    ax4.set_title('Reward vs Steps (Green=Success, Red=Failure)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved summary plot to: {save_path}")

    return fig

def main():
    # Configuration
    checkpoint_path = 'ppo_3m_output/checkpoint_3004416.pt'
    output_dir = './visualizations_3m'
    n_episodes = 10
    device = torch.device('cpu')

    os.makedirs(output_dir, exist_ok=True)

    print('=' * 80)
    print('VISUALIZING 3M PPO POLICY')
    print('=' * 80)
    print()
    print(f'Checkpoint: {checkpoint_path}')
    print(f'Episodes to visualize: {n_episodes}')
    print(f'Output directory: {output_dir}')
    print()

    # Load policy
    print('Loading policy...')
    policy, timesteps = load_policy(checkpoint_path, device)
    print(f'Loaded policy trained for {timesteps:,} timesteps')
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

    # Run episodes
    print('=' * 80)
    print(f'RUNNING {n_episodes} EPISODES')
    print('=' * 80)
    print()

    results = run_multiple_episodes(env, policy, n_episodes=n_episodes,
                                     deterministic=True, device=device)

    # Print summary
    successes = sum(1 for r in results if r['goal_reached'])
    total_rewards = [r['total_reward'] for r in results]
    steps = [r['steps'] for r in results]

    print()
    print('=' * 80)
    print('EVALUATION SUMMARY')
    print('=' * 80)
    print(f'Success Rate: {successes}/{n_episodes} ({100*successes/n_episodes:.1f}%)')
    print(f'Average Total Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}')
    print(f'Average Steps: {np.mean(steps):.1f} ± {np.std(steps):.1f}')
    print()

    # Generate visualizations
    print('=' * 80)
    print('GENERATING VISUALIZATIONS')
    print('=' * 80)
    print()

    # Visualize individual episodes
    for i, result in enumerate(results):
        if i < 5:  # Only visualize first 5 episodes
            print(f'Creating visualization for episode {i+1}...')
            save_path = os.path.join(output_dir, f'episode_{i+1:02d}_{"success" if result["goal_reached"] else "failure"}.png')

            # Temporarily set environment to this episode's state
            # We'll recreate a new env with the same seed
            vis_env = OccupancyGridEnv(
                world_width=10.0,
                world_height=10.0,
                num_static_obstacles=5,
                num_dynamic_obstacles=2,
                max_episode_steps=500,
                random_seed=42 + i + 1000,  # Same seed used during run
            )

            visualize_episode(vis_env, result, episode_num=i+1, save_path=save_path)
            plt.close()

    # Generate summary statistics plot
    print('Creating summary statistics plot...')
    summary_path = os.path.join(output_dir, 'summary_statistics.png')
    plot_summary_statistics(results, save_path=summary_path)
    plt.close()

    print()
    print('=' * 80)
    print('VISUALIZATION COMPLETE')
    print('=' * 80)
    print(f'All visualizations saved to: {output_dir}/')
    print()
    print('Generated files:')
    for f in sorted(os.listdir(output_dir)):
        if f.endswith('.png'):
            print(f'  - {f}')
    print()

if __name__ == '__main__':
    main()
