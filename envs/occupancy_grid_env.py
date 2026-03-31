"""
Gymnasium Environment for Occupancy Grid Navigation

This environment provides:
- 2D occupancy grid observation (32x32 local view)
- Robot pose, target relative position, and velocity observations
- Continuous action space for holonomic control
- Dense + sparse reward function
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, Dict, Any, List
import warnings

from .grid_world import GridWorld, CellType
from .robot_kinematics import HolonomicKinematics, RobotConfig


class OccupancyGridEnv(gym.Env):
    """
    Gymnasium environment for holonomic robot navigation in occupancy grid.

    Observation Space (Dict):
        - occupancy_grid: (32, 32) float32 [0, 1] - local view around robot
        - robot_pose: (3,) float32 [x, y, theta] - world pose
        - target_relative: (2,) float32 [dx, dy] - normalized relative to robot
        - velocity: (3,) float32 [vx, vy, omega] - body velocities

    Action Space (Box):
        - [vx, vy, omega] continuous
        - vx: [-0.5, 0.5] m/s (forward velocity)
        - vy: [-0.5, 0.5] m/s (lateral velocity)
        - omega: [-90, 90] deg/s (angular velocity)

    Reward Function:
        - Step penalty: -0.01 per step
        - Distance improvement: 10.0 * (prev_dist - curr_dist)
        - Collision penalty: -0.1
        - Goal bonus: +100.0
        - Action magnitude penalty: -0.001 * |action|
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        world_width: float = 10.0,
        world_height: float = 10.0,
        grid_resolution: float = 0.3125,  # 32x32 for 10x10m
        num_static_obstacles: int = 5,
        num_dynamic_obstacles: int = 2,
        max_episode_steps: int = 500,
        goal_threshold: float = 0.5,
        collision_threshold: float = 0.3,
        random_seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        hard_scenario: bool = False,
    ):
        """
        Initialize the environment.

        Args:
            world_width: World width in meters
            world_height: World height in meters
            grid_resolution: Grid cell size in meters
            num_static_obstacles: Number of static obstacles
            num_dynamic_obstacles: Number of dynamic obstacles
            max_episode_steps: Maximum steps per episode
            goal_threshold: Distance threshold for goal reached
            collision_threshold: Robot radius for collision checking
            random_seed: Random seed for reproducibility
            render_mode: Rendering mode ("human" or "rgb_array")
            hard_scenario: If True, generate structured complex obstacles
                that require detouring to reach the goal
        """
        super().__init__()

        # Environment parameters
        self.world_width = world_width
        self.world_height = world_height
        self.grid_resolution = grid_resolution
        self.max_episode_steps = max_episode_steps
        self.goal_threshold = goal_threshold
        self.collision_threshold = collision_threshold
        self.random_seed = random_seed
        self.render_mode = render_mode
        self.hard_scenario = hard_scenario

        # Create grid world
        self.grid_world = GridWorld(
            width=world_width,
            height=world_height,
            grid_resolution=grid_resolution,
            num_static_obstacles=num_static_obstacles,
            num_dynamic_obstacles=num_dynamic_obstacles,
            random_seed=random_seed,
        )

        # Generate hard scenario with structured obstacles if requested
        if hard_scenario:
            self.grid_world.generate_hard_scenario()

        # Create kinematics model
        self.kinematics = HolonomicKinematics(RobotConfig())

        # State variables
        self.robot_state = None  # [x, y, theta, vx, vy, omega]
        self.prev_distance_to_goal = None
        self.episode_step = 0
        self.episode_reward = 0.0

        # Define observation space (Dict)
        self.observation_space = spaces.Dict({
            "occupancy_grid": spaces.Box(
                low=0.0, high=1.0,
                shape=(32, 32),
                dtype=np.float32
            ),
            "robot_pose": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(3,),
                dtype=np.float32
            ),
            "target_relative": spaces.Box(
                low=-1.0, high=1.0,
                shape=(2,),
                dtype=np.float32
            ),
            "velocity": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(3,),
                dtype=np.float32
            ),
        })

        # Define action space (continuous)
        # [vx, vy, omega] in m/s, m/s, deg/s
        self.action_space = spaces.Box(
            low=np.array([-0.5, -0.5, -90.0], dtype=np.float32),
            high=np.array([0.5, 0.5, 90.0], dtype=np.float32),
            dtype=np.float32
        )

        # Rendering
        self._render_cache = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Returns:
            observation: Initial observation dict
            info: Additional information dict
        """
        # Set seed if provided
        if seed is not None:
            self.random_seed = seed
            self.grid_world.rng = np.random.RandomState(seed)

        # Reset grid world
        self.grid_world.reset(randomize=True)

        # If hard scenario, regenerate the obstacle layout
        if self.hard_scenario:
            self.grid_world.generate_hard_scenario()

        # Sample robot start position
        robot_pos = self.grid_world.sample_free_position(
            min_clearance=self.collision_threshold + 0.2
        )
        if robot_pos is None:
            # Fallback to center if sampling fails
            robot_pos = np.array([self.world_width / 2, self.world_height / 2])

        # Sample goal position (far from robot)
        # For hard scenarios, ensure straight-line path is blocked
        for attempt in range(100):
            goal_pos = self.grid_world.sample_free_position(
                min_clearance=self.collision_threshold + 0.2
            )
            if goal_pos is not None:
                dist = np.linalg.norm(goal_pos - robot_pos)
                if dist > 3.0:  # At least 3 meters away
                    # For hard scenarios, check that straight-line is blocked
                    if not self.hard_scenario or self.grid_world.is_straight_line_blocked(robot_pos, goal_pos):
                        break

        if goal_pos is None:
            # Fallback
            goal_pos = np.array([self.world_width - 1.0, self.world_height - 1.0])

        self.grid_world.set_goal(goal_pos)

        # Initialize robot state: [x, y, theta, vx, vy, omega]
        theta = np.arctan2(
            goal_pos[1] - robot_pos[1],
            goal_pos[0] - robot_pos[0]
        )
        self.robot_state = np.array([
            robot_pos[0], robot_pos[1], theta,
            0.0, 0.0, 0.0  # Initial velocities are zero
        ])

        # Track distance for reward
        self.prev_distance_to_goal = np.linalg.norm(robot_pos - goal_pos)

        # Reset step counter
        self.episode_step = 0
        self.episode_reward = 0.0

        # Build observation
        observation = self._get_observation()

        # Build info
        info = {
            "distance_to_goal": self.prev_distance_to_goal,
            "robot_position": robot_pos.copy(),
            "goal_position": goal_pos.copy(),
        }

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[
        Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]
    ]:
        """
        Execute one time step in the environment.

        Args:
            action: [vx, vy, omega] velocity command

        Returns:
            observation: Current observation
            reward: Reward for this step
            terminated: Whether episode ended (success/failure)
            truncated: Whether episode was truncated (timeout)
            info: Additional information
        """
        # Validate action
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Time step
        dt = 0.1  # 10 Hz control

        # Update dynamic obstacles
        self.grid_world.update_dynamic_obstacles(dt)

        # Integrate robot dynamics
        body_velocity = np.array([
            action[0],  # vx
            action[1],  # vy
            np.deg2rad(action[2])  # omega (convert deg/s to rad/s)
        ])

        self.robot_state = self.kinematics.integrate_state(
            self.robot_state,
            body_velocity,
            dt
        )

        # Check for collisions
        robot_pos = self.robot_state[:2]
        collision = self.grid_world.check_collision(
            robot_pos,
            robot_radius=self.collision_threshold
        )

        # Check goal reached
        goal_pos = self.grid_world.goal_position
        distance_to_goal = np.linalg.norm(robot_pos - goal_pos)
        goal_reached = distance_to_goal < self.goal_threshold

        # Compute reward
        reward = self._compute_reward(
            action=action,
            distance_to_goal=distance_to_goal,
            collision=collision,
            goal_reached=goal_reached
        )

        # Update tracking
        self.prev_distance_to_goal = distance_to_goal
        self.episode_step += 1
        self.episode_reward += reward

        # Check termination conditions
        terminated = collision or goal_reached
        truncated = self.episode_step >= self.max_episode_steps

        # Build observation
        observation = self._get_observation()

        # Build info
        info = {
            "distance_to_goal": distance_to_goal,
            "collision": collision,
            "goal_reached": goal_reached,
            "episode_step": self.episode_step,
            "episode_reward": self.episode_reward,
        }

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Build observation dictionary from current state."""
        # Extract state components
        x, y, theta = self.robot_state[0], self.robot_state[1], self.robot_state[2]
        vx, vy, omega = self.robot_state[3], self.robot_state[4], self.robot_state[5]

        # Get local occupancy grid
        robot_pos = np.array([x, y])
        occupancy_grid = self.grid_world.get_occupancy_grid_at_position(
            robot_pos,
            robot_radius=self.collision_threshold
        )

        # Robot pose
        robot_pose = np.array([x, y, theta], dtype=np.float32)

        # Target relative position (in robot frame)
        if self.grid_world.goal_position is not None:
            goal_pos = self.grid_world.goal_position
            dx_world = goal_pos[0] - x
            dy_world = goal_pos[1] - y

            # Transform to robot frame
            cos_theta = np.cos(-theta)
            sin_theta = np.sin(-theta)
            dx_robot = dx_world * cos_theta - dy_world * sin_theta
            dy_robot = dx_world * sin_theta + dy_world * cos_theta

            # Normalize by world size for stable learning
            max_dist = np.sqrt(self.world_width**2 + self.world_height**2)
            target_relative = np.array([
                dx_robot / max_dist,
                dy_robot / max_dist
            ], dtype=np.float32)
        else:
            target_relative = np.zeros(2, dtype=np.float32)

        # Velocity (already in robot frame)
        velocity = np.array([vx, vy, omega], dtype=np.float32)

        return {
            "occupancy_grid": occupancy_grid,
            "robot_pose": robot_pose,
            "target_relative": target_relative,
            "velocity": velocity,
        }

    def _compute_reward(
        self,
        action: np.ndarray,
        distance_to_goal: float,
        collision: bool,
        goal_reached: bool
    ) -> float:
        """
        Compute reward for current step.

        Reward components:
        - Step penalty: small constant penalty per step
        - Distance improvement: reward for getting closer to goal
        - Collision penalty: large negative reward for collisions
        - Goal bonus: large positive reward for reaching goal
        - Action penalty: small penalty for large actions (regularization)
        """
        reward = 0.0

        # Step penalty (encourage efficiency)
        reward += -0.01

        # Distance improvement (dense reward)
        if self.prev_distance_to_goal is not None:
            distance_improvement = self.prev_distance_to_goal - distance_to_goal
            reward += 10.0 * distance_improvement

        # Collision penalty
        if collision:
            reward += -10.0

        # Goal reached bonus (sparse reward)
        if goal_reached:
            reward += 100.0

        # Action magnitude penalty (regularization)
        action_magnitude = np.linalg.norm(action)
        reward += -0.001 * action_magnitude

        return float(reward)

    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.

        Returns:
            RGB image array if render_mode is "rgb_array", else None
        """
        if self.robot_state is None:
            return None

        robot_pos = self.robot_state[:2]
        image = self.grid_world.render(robot_pos)

        if self.render_mode == "rgb_array":
            return (image * 255).astype(np.uint8)
        elif self.render_mode == "human":
            # Would display with pygame or similar
            return None

        return None

    def close(self):
        """Close the environment and cleanup resources."""
        pass

    def get_metrics(self) -> Dict[str, float]:
        """Get current environment metrics for logging."""
        if self.robot_state is None:
            return {}

        robot_pos = self.robot_state[:2]
        distance = 0.0
        if self.grid_world.goal_position is not None:
            distance = np.linalg.norm(robot_pos - self.grid_world.goal_position)

        return {
            "distance_to_goal": distance,
            "episode_step": self.episode_step,
            "episode_reward": self.episode_reward,
        }
