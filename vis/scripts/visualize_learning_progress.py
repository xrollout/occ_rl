"""
Visualize learning progress: success rate and mean reward vs training steps from 300k to 5M.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Results from evaluation
results = [
    {
        'timesteps': 301056,
        'timesteps_label': '301k',
        'success_rate': 15.00,
        'mean_reward': 31.80
    },
    {
        'timesteps': 1001472,
        'timesteps_label': '1M',
        'success_rate': 17.00,
        'mean_reward': 31.09
    },
    {
        'timesteps': 3002368,
        'timesteps_label': '3M',
        'success_rate': 42.00,
        'mean_reward': 65.48
    },
    {
        'timesteps': 5003264,
        'timesteps_label': '5M',
        'success_rate': 62.00,
        'mean_reward': 92.08
    },
    {
        'timesteps': 10004480,
        'timesteps_label': '10M',
        'success_rate': 15.00,
        'mean_reward': 24.24
    }
]

timesteps = [r['timesteps'] / 1e6 for r in results]
success_rates = [r['success_rate'] for r in results]
mean_rewards = [r['mean_reward'] for r in results]
labels = [r['timesteps_label'] for r in results]

def main():
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Success rate plot
    ax1.plot(timesteps, success_rates, 'o-', linewidth=3, markersize=10, color='#FF6B6B', alpha=0.8)
    ax1.set_xlabel('Training Steps (Millions)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Success Rate vs Training Steps', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 70)

    # Add value labels on points
    for i, (x, y) in enumerate(zip(timesteps, success_rates)):
        ax1.text(x, y + 2, f'{y}%', ha='center', fontsize=11, fontweight='bold')

    # Set xticks to match our points
    ax1.set_xticks(timesteps)
    ax1.set_xticklabels(labels)

    # Mean reward plot
    ax2.plot(timesteps, mean_rewards, 's-', linewidth=3, markersize=8, color='#4ECDC4', alpha=0.8)
    ax2.set_xlabel('Training Steps (Millions)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Reward per Episode', fontsize=12, fontweight='bold')
    ax2.set_title('Mean Reward vs Training Steps', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)

    # Add value labels on points
    for i, (x, y) in enumerate(zip(timesteps, mean_rewards)):
        ax2.text(x, y + 3, f'{y:.2f}', ha='center', fontsize=11, fontweight='bold')

    # Set xticks to match our points
    ax2.set_xticks(timesteps)
    ax2.set_xticklabels(labels)

    # Main title
    fig.suptitle('PPO Learning Progress on Obstacle Navigation Task\n(includes collapse at 10M with fixed LR)', fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save
    os.makedirs('./output/learning_curves', exist_ok=True)
    save_path = './output/learning_curves/learning_progress.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f'Saved learning progress visualization to: {save_path}')

    # Also save a text summary
    summary_path = './output/learning_curves/progress_summary.txt'
    with open(summary_path, 'w') as f:
        f.write('PPO Learning Progress Summary\n')
        f.write('==============================\n\n')
        f.write(f'{"Training Steps":<15} {"Success Rate":<15} {"Mean Reward":<15}\n')
        f.write(f'{"-"*45}\n')
        for r in results:
            f.write(f'{r["timesteps_label"]:<15} {r["success_rate"]:>11.2f}%       {r["mean_reward"]:>10.2f}\n')
    print(f'Saved summary to: {summary_path}')

    plt.show()

if __name__ == '__main__':
    main()
