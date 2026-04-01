"""
Plot success rate vs training steps.
"""
import os
import matplotlib.pyplot as plt

# Data from all evaluations
data = [
    {'steps': 301056, 'success_rate': 9.0, 'label': '300k (constant ent)'},
    {'steps': 1001472, 'success_rate': 17.0, 'label': '1M (constant ent)'},
    {'steps': 3002368, 'success_rate': 47.0, 'label': '3M (constant ent)'},
    {'steps': 5003264, 'success_rate': 62.0, 'label': '5M (constant ent)'},
    {'steps': 10004480, 'success_rate': 5.0, 'label': '10M (constant ent)'},
    {'steps': 10004480, 'success_rate': 49.0, 'label': '10M (entropy annealing)'},
]

# Separate constant entropy vs annealing
constant_data = [d for d in data if 'constant ent' in d['label']]
annealing_data = [d for d in data if 'annealing' in d['label']]

# Create plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot constant entropy
constant_steps = [d['steps'] / 1_000_000 for d in constant_data]
constant_rates = [d['success_rate'] for d in constant_data]
ax.plot(constant_steps, constant_rates, 'bo-', linewidth=2, markersize=8, label='Constant entropy (0.01)')

# Plot annealing point
if annealing_data:
    anneal_steps = [d['steps'] / 1_000_000 for d in annealing_data]
    anneal_rates = [d['success_rate'] for d in annealing_data]
    ax.scatter(anneal_steps, anneal_rates, color='red', s=150, marker='*', zorder=5, label='Entropy annealing (0.01 → 0.001)')

ax.set_xlabel('Training Steps (millions)', fontsize=12)
ax.set_ylabel('Success Rate (%)', fontsize=12)
ax.set_title('Success Rate vs Training Steps for PPO on Navigation Task', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
ax.set_ylim(0, 70)

# Add annotations for key points
for d in data:
    x = d['steps'] / 1_000_000
    y = d['success_rate']
    ax.annotate(f"{y}%", (x, y), xytext=(5, 5), textcoords='offset points', fontsize=10)

plt.tight_layout()

# Save
os.makedirs('./output/plots', exist_ok=True)
save_path = './output/plots/success_rate_vs_steps.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Saved plot to: {save_path}")
plt.show()
