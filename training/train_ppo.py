"""
PPO Training Script for Grid Navigation

This script trains a PPO policy for holonomic robot navigation using RLlib.

Usage:
    # Basic training
    python train_ppo.py --exp-name=grid_nav_v1

    # With custom config
    python train_ppo.py --exp-name=grid_nav_v2 \
        --num-envs=8 \
        --lr=3e-4 \
        --num-iterations=500

    # Resume from checkpoint
    python train_ppo.py --exp-name=grid_nav_v1 \
        --resume=/path/to/checkpoint

Features:
- Custom CNN+MLP policy architecture for grid navigation
- Vectorized environment support for parallel training
- Automatic checkpointing and logging
- TensorBoard integration
- Curriculum learning support (easy -> hard)
"""

import argparse
import os
import sys
import json
from datetime import datetime
from typing import Dict, Optional

import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train PPO policy for grid navigation"
    )

    # Experiment settings
    parser.add_argument(
        "--exp-name",
        type=str,
        default=f"grid_nav_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Experiment name for logging"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="~/ray_results/grid_nav",
        help="Output directory for checkpoints and logs"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    # Training settings
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=1000,
        help="Number of training iterations"
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="Number of parallel environments"
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=None,
        help="Number of CPUs to use (default: auto)"
    )
    parser.add_argument(
        "--num-gpus",
        type=float,
        default=0,
        help="Number of GPUs to use"
    )

    # PPO hyperparameters
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor"
    )
    parser.add_argument(
        "--lambda",
        type=float,
        default=0.95,
        help="GAE lambda"
    )
    parser.add_argument(
        "--clip-param",
        type=float,
        default=0.2,
        help="PPO clip parameter"
    )
    parser.add_argument(
        "--entropy-coeff",
        type=float,
        default=0.01,
        help="Entropy coefficient"
    )
    parser.add_argument(
        "--vf-loss-coeff",
        type=float,
        default=0.5,
        help="Value function loss coefficient"
    )

    # Environment settings
    parser.add_argument(
        "--world-size",
        type=float,
        default=10.0,
        help="World size in meters"
    )
    parser.add_argument(
        "--num-static-obstacles",
        type=int,
        default=5,
        help="Number of static obstacles"
    )
    parser.add_argument(
        "--num-dynamic-obstacles",
        type=int,
        default=2,
        help="Number of dynamic obstacles"
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Enable curriculum learning"
    )

    # Checkpointing
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint path"
    )
    parser.add_argument(
        "--pretrained-ckpt",
        type=str,
        default=None,
        help="Path to pretrained behavior cloning checkpoint to initialize from"
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=50,
        help="Checkpoint frequency (iterations)"
    )
    parser.add_argument(
        "--checkpoint-at-end",
        action="store_true",
        default=True,
        help="Save checkpoint at end of training"
    )

    # Evaluation
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=50,
        help="Evaluation frequency (iterations)"
    )
    parser.add_argument(
        "--eval-duration",
        type=int,
        default=10,
        help="Number of episodes per evaluation"
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--framework",
        type=str,
        default="torch",
        choices=["torch", "tf2"],
        help="Deep learning framework"
    )

    return parser.parse_args()


def setup_config(args) -> Dict:
    """
    Create RLlib configuration from command line arguments.

    This function creates a complete PPO configuration dictionary
    with all necessary settings for training the grid navigation policy.
    """
    # Import here to avoid dependency issues when just parsing args
    try:
        from ray.rllib.models import ModelCatalog
        from ray.rllib.utils.framework import try_import_torch
        torch, _ = try_import_torch()
    except ImportError:
        print("Warning: Ray/RLlib not installed. Configuration will be incomplete.")
        torch = None

    # Environment configuration
    env_config = {
        "world_width": args.world_size,
        "world_height": args.world_size,
        "grid_resolution": 0.3125,  # 32x32 for 10x10m
        "num_static_obstacles": args.num_static_obstacles,
        "num_dynamic_obstacles": args.num_dynamic_obstacles,
        "max_episode_steps": 500,
        "goal_threshold": 0.5,
        "collision_threshold": 0.3,
        "random_seed": args.seed,
        "render_mode": None,
    }

    # Model configuration
    model_config = {
        "custom_model": "grid_nav_model",
        "custom_model_config": {
            "grid_size": 32,
            "grid_channels": 1,
            "grid_features": 256,
            "scalar_features": 64,
            "shared_features": 256,
        },
        "vf_share_layers": False,  # Separate value network
    }

    # PPO configuration
    config = {
        # Environment
        "env": "occupancy_grid_env",
        "env_config": env_config,

        # Framework
        "framework": args.framework,

        # Model
        "model": model_config,

        # Resources
        "num_workers": args.num_envs - 1 if args.num_envs > 1 else 0,
        "num_envs_per_worker": 1,
        "num_cpus_per_worker": 1,
        "num_gpus": args.num_gpus,
        "num_gpus_per_worker": 0,

        # Rollout settings
        "rollout_fragment_length": 200,
        "train_batch_size": 4000,

        # PPO specific settings
        "lr": args.lr,
        "gamma": args.gamma,
        "lambda": getattr(args, 'lambda'),  # GAE lambda
        "clip_param": args.clip_param,
        "entropy_coeff": args.entropy_coeff,
        "vf_loss_coeff": args.vf_loss_coeff,
        "use_gae": True,
        "use_critic": True,
        "grad_clip": 0.5,

        # Training settings
        "batch_mode": "truncate_episodes",
        "observation_filter": "MeanStdFilter",

        # Evaluation
        "evaluation_interval": args.eval_freq,
        "evaluation_duration": args.eval_duration,
        "evaluation_config": {
            "explore": False,
        },

        # Logging
        "log_level": args.log_level,
        "metrics_smoothing_episodes": 100,

        # Callbacks (for curriculum learning)
        "callbacks": GridNavCallbacks if args.curriculum else None,
    }

    return config


class GridNavCallbacks:
    """
    RLlib callbacks for curriculum learning and custom metrics.

    This callback class implements curriculum learning that gradually
    increases the difficulty of the environment as the policy improves.
    """

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        """
        Called after each training iteration.

        Updates curriculum difficulty based on training performance.
        """
        # Get current curriculum level (stored in algorithm config)
        current_level = algorithm.config.get("env_config", {}).get("curriculum_level", 0)

        # Get evaluation results
        if "evaluation" in result:
            eval_results = result["evaluation"]
            episode_reward_mean = eval_results.get("episode_reward_mean", float('-inf'))
            episode_len_mean = eval_results.get("episode_len_mean", float('inf'))

            # Curriculum criteria: good reward + reaching goal quickly
            if episode_reward_mean > 50 and episode_len_mean < 200:
                # Increase difficulty
                new_level = current_level + 1
                self._set_curriculum_level(algorithm, new_level)
                print(f"Curriculum: Increased to level {new_level}")

        # Log curriculum level
        result["curriculum_level"] = current_level

    def _set_curriculum_level(self, algorithm, level: int):
        """
        Update environment configuration for new curriculum level.

        Each level increases difficulty by:
        - More static obstacles
        - More dynamic obstacles
        - Smaller goal threshold
        """
        base_obstacles = 3
        base_dynamic = 1

        # Calculate new difficulty
        num_static = base_obstacles + level
        num_dynamic = base_dynamic + level // 2
        goal_threshold = max(0.3, 0.5 - level * 0.05)

        # Update environment config
        new_env_config = {
            "num_static_obstacles": num_static,
            "num_dynamic_obstacles": num_dynamic,
            "goal_threshold": goal_threshold,
            "curriculum_level": level,
        }

        # Apply to all workers
        algorithm.workers.foreach_env_with_context(
            lambda env, env_context: env_context.config.update(new_env_config)
        )

    def on_episode_start(self, *, worker, base_env, policies, episode, **kwargs):
        """Called at the start of each episode."""
        episode.user_data["distance_traveled"] = 0.0
        episode.user_data["prev_position"] = None

    def on_episode_step(self, *, worker, base_env, policies, episode, **kwargs):
        """Called at each step."""
        # Track distance traveled
        if episode.user_data["prev_position"] is not None:
            info = episode.last_info_for()
            if info and "robot_position" in info:
                curr_pos = np.array(info["robot_position"])
                prev_pos = episode.user_data["prev_position"]
                dist = np.linalg.norm(curr_pos - prev_pos)
                episode.user_data["distance_traveled"] += dist
                episode.user_data["prev_position"] = curr_pos

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        """Called at the end of each episode."""
        # Log custom metrics
        episode.custom_metrics["distance_traveled"] = episode.user_data.get("distance_traveled", 0)


def main():
    """Main training function."""
    args = parse_args()

    # Import Ray here to avoid dependency issues during arg parsing
    try:
        import ray
        from ray import tune
        from ray.rllib.algorithms.ppo import PPO
        from ray.rllib.models import ModelCatalog
        from ray.rllib.env.env_context import EnvContext
    except ImportError as e:
        print(f"Error: Ray/RLlib not installed. {e}")
        print("Install with: pip install ray[rllib]")
        return 1

    # Import custom components
    from envs import OccupancyGridEnv
    from policies.grid_nav_policy import GridNavTorchModel

    # Register custom model and environment
    ModelCatalog.register_custom_model("grid_nav_model", GridNavTorchModel)

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(
            num_cpus=args.num_cpus or args.num_envs + 2,
            num_gpus=args.num_gpus,
            log_to_driver=False,
        )

    # Setup configuration
    config = setup_config(args)

    # Update config with custom model registration
    config["env"] = OccupancyGridEnv

    # Create output directory
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    config_path = os.path.join(output_dir, f"{args.exp_name}_config.json")
    with open(config_path, 'w') as f:
        # Convert to JSON-serializable format
        json_config = {}
        for key, value in config.items():
            if key not in ['callbacks', 'env']:  # Skip non-serializable
                try:
                    json.dumps({key: value})
                    json_config[key] = value
                except (TypeError, ValueError):
                    pass
        json.dump(json_config, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Training PPO for Grid Navigation")
    print(f"{'='*70}")
    print(f"Experiment: {args.exp_name}")
    print(f"Output directory: {output_dir}")
    print(f"Config saved to: {config_path}")
    print(f"\nTraining settings:")
    print(f"  - Iterations: {args.num_iterations}")
    print(f"  - Parallel envs: {args.num_envs}")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Gamma: {args.gamma}")
    print(f"  - Lambda: {getattr(args, 'lambda')}")
    print(f"{'='*70}\n")

    # Create algorithm
    algo = PPO(config=config)

    # Load pretrained behavior cloning checkpoint if provided
    if args.pretrained_ckpt:
        print(f"Loading pretrained behavior cloning checkpoint: {args.pretrained_ckpt}")
        import torch
        ckpt = torch.load(args.pretrained_ckpt, map_location='cpu', weights_only=False)

        # Get the RLlib model
        policy = algo.get_policy()
        model = policy.model

        # Load weights from BC (ActorCriticPolicy) to RLlib (GridNavTorchModel)
        # The architectures are similar but have different naming conventions
        bc_state = ckpt['policy_state_dict']

        # Create mapping from BC names to RLlib names
        name_mapping = {
            'grid_encoder.0.weight': 'grid_encoder.conv.0.weight',
            'grid_encoder.0.bias': 'grid_encoder.conv.0.bias',
            'grid_encoder.2.weight': 'grid_encoder.conv.2.weight',
            'grid_encoder.2.bias': 'grid_encoder.conv.2.bias',
            'grid_encoder.5.weight': 'grid_encoder.fc.0.weight',
            'grid_encoder.5.bias': 'grid_encoder.fc.0.bias',
            'scalar_encoder.0.weight': 'scalar_encoder.mlp.0.weight',
            'scalar_encoder.0.bias': 'scalar_encoder.mlp.0.bias',
            'shared.0.weight': 'shared_fc.0.weight',
            'shared.0.bias': 'shared_fc.0.bias',
            'shared.2.weight': 'shared_fc.2.weight',
            'shared.2.bias': 'shared_fc.2.bias',
        }

        # Load the mapped weights
        model_state = model.state_dict()
        loaded_count = 0

        for bc_key, rllib_key in name_mapping.items():
            if bc_key in bc_state and rllib_key in model_state:
                model_state[rllib_key] = bc_state[bc_key]
                loaded_count += 1

        print(f"Loaded {loaded_count} layers from pretrained BC checkpoint")
        model.load_state_dict(model_state)
        print(f"Successfully initialized model with pretrained BC weights")

    # Load checkpoint if resuming
    start_iteration = 0
    if args.resume:
        print(f"Restoring from checkpoint: {args.resume}")
        algo.restore(args.resume)
        # Extract iteration number from checkpoint path if possible
        try:
            start_iteration = int(args.resume.split('_')[-1])
        except ValueError:
            pass

    # Training loop
    print("Starting training...\n")
    try:
        for iteration in range(start_iteration, args.num_iterations):
            # Train one iteration
            result = algo.train()

            # Print progress
            current_iteration = iteration + 1
            if current_iteration % 10 == 0 or current_iteration == 1:
                episode_reward_mean = result.get('episode_reward_mean', float('nan'))
                episode_len_mean = result.get('episode_len_mean', float('nan'))
                total_steps = result.get('timesteps_total', 0)

                print(f"Iter {current_iteration:4d}/{args.num_iterations}: "
                      f"reward={episode_reward_mean:8.2f}, "
                      f"len={episode_len_mean:6.1f}, "
                      f"steps={total_steps:8d}")

            # Save checkpoint
            if args.checkpoint_freq > 0 and current_iteration % args.checkpoint_freq == 0:
                checkpoint_dir = algo.save(os.path.join(output_dir, args.exp_name))
                print(f"  Checkpoint saved: {checkpoint_dir}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    finally:
        # Final checkpoint
        if args.checkpoint_at_end:
            checkpoint_dir = algo.save(os.path.join(output_dir, args.exp_name + "_final"))
            print(f"\nFinal checkpoint saved: {checkpoint_dir}")

        # Clean up
        algo.stop()
        ray.shutdown()

    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"{'='*70}")
    print(f"Experiment: {args.exp_name}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
