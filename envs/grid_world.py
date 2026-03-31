"""
2D Occupancy Grid World Simulation

Provides a configurable grid world with:
- Static obstacles (walls, random blocks)
- Dynamic obstacles (moving agents)
- Goal/target positions
- Collision detection
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
from enum import IntEnum


class CellType(IntEnum):
    """Cell type enumeration for grid visualization."""
    FREE = 0
    OCCUPIED = 1
    ROBOT = 2
    GOAL = 3
    DYNAMIC_OBSTACLE = 4


@dataclass
class Obstacle:
    """Base class for obstacles."""
    position: np.ndarray  # [x, y]
    radius: float = 0.5

    def get_grid_cells(self, grid_resolution: float) -> List[Tuple[int, int]]:
        """Get grid cells occupied by this obstacle."""
        cells = []
        radius_cells = int(np.ceil(self.radius / grid_resolution))
        center_i = int(self.position[1] / grid_resolution)
        center_j = int(self.position[0] / grid_resolution)

        for di in range(-radius_cells, radius_cells + 1):
            for dj in range(-radius_cells, radius_cells + 1):
                dist = np.sqrt(di**2 + dj**2) * grid_resolution
                if dist <= self.radius:
                    cells.append((center_i + di, center_j + dj))

        return cells


@dataclass
class StaticObstacle(Obstacle):
    """Static obstacle that doesn't move."""
    pass


@dataclass
class DynamicObstacle(Obstacle):
    """Dynamic obstacle that moves with constant velocity."""
    velocity: np.ndarray = None  # [vx, vy]
    bounds: Tuple[float, float, float, float] = None  # [xmin, xmax, ymin, ymax]

    def __post_init__(self):
        if self.velocity is None:
            self.velocity = np.zeros(2)

    def update(self, dt: float):
        """Update position based on velocity with bouncing at boundaries."""
        new_position = self.position + self.velocity * dt

        if self.bounds is not None:
            xmin, xmax, ymin, ymax = self.bounds

            # Bounce off x boundaries
            if new_position[0] < xmin + self.radius or new_position[0] > xmax - self.radius:
                self.velocity[0] *= -1
                new_position[0] = np.clip(new_position[0], xmin + self.radius, xmax - self.radius)

            # Bounce off y boundaries
            if new_position[1] < ymin + self.radius or new_position[1] > ymax - self.radius:
                self.velocity[1] *= -1
                new_position[1] = np.clip(new_position[1], ymin + self.radius, ymax - self.radius)

        self.position = new_position


class GridWorld:
    """
    2D Occupancy Grid World for robot navigation.

    The world is defined by:
    - width, height: Physical dimensions in meters
    - grid_resolution: Size of each grid cell in meters
    - Obstacles (static and dynamic)
    - Goal position

    Grid coordinates: (i, j) where i is row (y-axis), j is column (x-axis)
    World coordinates: (x, y) where x is horizontal, y is vertical
    """

    def __init__(
        self,
        width: float = 10.0,
        height: float = 10.0,
        grid_resolution: float = 0.3125,  # 32x32 grid for 10x10m
        num_static_obstacles: int = 5,
        num_dynamic_obstacles: int = 2,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize the grid world.

        Args:
            width: World width in meters
            height: World height in meters
            grid_resolution: Grid cell size in meters
            num_static_obstacles: Number of static obstacles
            num_dynamic_obstacles: Number of dynamic obstacles
            random_seed: Random seed for reproducibility
        """
        self.width = width
        self.height = height
        self.grid_resolution = grid_resolution

        # Calculate grid dimensions
        self.grid_height = int(np.ceil(height / grid_resolution))
        self.grid_width = int(np.ceil(width / grid_resolution))

        # Initialize random state
        self.rng = np.random.RandomState(random_seed)

        # Initialize occupancy grid
        self.occupancy_grid = np.zeros(
            (self.grid_height, self.grid_width), dtype=np.float32
        )

        # Obstacle lists
        self.static_obstacles: List[StaticObstacle] = []
        self.dynamic_obstacles: List[DynamicObstacle] = []

        # Goal position
        self.goal_position: Optional[np.ndarray] = None

        # Generate world
        self._generate_walls()
        self._generate_static_obstacles(num_static_obstacles)
        self._generate_dynamic_obstacles(num_dynamic_obstacles)

    def _world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates (x, y) to grid indices (i, j)."""
        j = int(x / self.grid_resolution)
        i = int(y / self.grid_resolution)
        return i, j

    def _grid_to_world(self, i: int, j: int) -> Tuple[float, float]:
        """Convert grid indices (i, j) to world coordinates (x, y)."""
        x = (j + 0.5) * self.grid_resolution
        y = (i + 0.5) * self.grid_resolution
        return x, y

    def _generate_walls(self):
        """Generate boundary walls around the world."""
        # Mark boundary cells as occupied
        self.occupancy_grid[0, :] = 1.0  # Bottom wall
        self.occupancy_grid[-1, :] = 1.0  # Top wall
        self.occupancy_grid[:, 0] = 1.0  # Left wall
        self.occupancy_grid[:, -1] = 1.0  # Right wall

    def _generate_static_obstacles(self, num_obstacles: int):
        """Generate random static obstacles."""
        for _ in range(num_obstacles):
            # Random position (avoiding walls)
            x = self.rng.uniform(1.0, self.width - 1.0)
            y = self.rng.uniform(1.0, self.height - 1.0)

            # Random radius
            radius = self.rng.uniform(0.3, 0.8)

            obstacle = StaticObstacle(
                position=np.array([x, y]),
                radius=radius
            )

            self.static_obstacles.append(obstacle)

            # Update occupancy grid
            cells = obstacle.get_grid_cells(self.grid_resolution)
            for i, j in cells:
                if 0 <= i < self.grid_height and 0 <= j < self.grid_width:
                    self.occupancy_grid[i, j] = 1.0

    def generate_hard_scenario(self):
        """
        Generate a hard navigation scenario that requires detouring around obstacles.

        This method creates structured obstacle layouts where the straight-line
        path from start to goal is guaranteed to be blocked. The robot must
        navigate around the obstacle(s) to reach the goal.

        Types of scenarios generated (randomly chosen):
        1. Central barrier - Large horizontal barrier across the middle
        2. U-shaped trap - U obstacle with goal inside the U
        3. Chained corridors - Multiple obstacles forcing zig-zag
        4. Cluttered with blocked straight-line - Random clutter that blocks direct path
        """
        # Clear existing static obstacles
        self.static_obstacles = []
        self.occupancy_grid[1:self.grid_height-1, 1:self.grid_width-1] = 0.0

        scenario_type = self.rng.randint(0, 4)

        if scenario_type == 0:
            self._generate_central_barrier()
        elif scenario_type == 1:
            self._generate_u_shaped_trap()
        elif scenario_type == 2:
            self._generate_chained_corridors()
        else:
            self._generate_cluttered_blocked()

        # Update occupancy grid for all obstacles
        for obstacle in self.static_obstacles:
            cells = obstacle.get_grid_cells(self.grid_resolution)
            for i, j in cells:
                if 0 <= i < self.grid_height and 0 <= j < self.grid_width:
                    self.occupancy_grid[i, j] = 1.0

    def _generate_central_barrier(self):
        """Generate a large horizontal barrier across the center of the map."""
        # Barrier across the middle, leaving openings on left and right
        center_y = self.height / 2
        barrier_length = self.width * 0.6
        barrier_x_start = self.width * 0.2
        barrier_radius = 0.6  # Width of barrier

        # Place multiple obstacles to form a continuous barrier
        num_obstacles = int(barrier_length / 1.0)
        for i in range(num_obstacles):
            x = barrier_x_start + i * 1.0
            y = center_y + self.rng.uniform(-0.2, 0.2)
            obstacle = StaticObstacle(
                position=np.array([x, y]),
                radius=barrier_radius
            )
            self.static_obstacles.append(obstacle)

    def _generate_u_shaped_trap(self):
        """Generate a U-shaped obstacle with opening on one side."""
        # U placed in center, goal inside the U
        center_x = self.width / 2
        center_y = self.height / 2

        # Three walls forming U shape
        # Left vertical wall
        obstacle_left = StaticObstacle(
            position=np.array([center_x - 1.5, center_y]),
            radius=2.0
        )
        # Bottom horizontal wall
        obstacle_bottom = StaticObstacle(
            position=np.array([center_x, center_y - 1.5]),
            radius=1.5
        )
        # Right vertical wall
        obstacle_right = StaticObstacle(
            position=np.array([center_x + 1.5, center_y]),
            radius=2.0
        )

        self.static_obstacles.extend([obstacle_left, obstacle_bottom, obstacle_right])

    def _generate_chained_corridors(self):
        """Generate multiple alternating obstacles that force zig-zag navigation."""
        # Alternating barriers on left and right, forcing the robot to weave
        num_pairs = 3
        spacing = self.width / (num_pairs + 1)

        for i in range(num_pairs):
            if i % 2 == 0:
                # Barrier on the left
                x = spacing * (i + 1)
                y = self.height / 2
                obstacle = StaticObstacle(
                    position=np.array([x, y]),
                    radius=2.0
                )
            else:
                # Barrier on the right
                x = spacing * (i + 1)
                y = self.height / 2
                obstacle = StaticObstacle(
                    position=np.array([x, y]),
                    radius=2.0
                )
            self.static_obstacles.append(obstacle)

    def _generate_cluttered_blocked(self):
        """Generate random clutter but guarantee straight-line is blocked."""
        # Add 6-8 random obstacles
        num_obstacles = self.rng.randint(6, 9)
        for _ in range(num_obstacles):
            x = self.rng.uniform(1.5, self.width - 1.5)
            y = self.rng.uniform(1.5, self.height - 1.5)
            radius = self.rng.uniform(0.5, 1.0)
            obstacle = StaticObstacle(
                position=np.array([x, y]),
                radius=radius
            )
            self.static_obstacles.append(obstacle)

    def is_straight_line_blocked(self, start: np.ndarray, goal: np.ndarray) -> bool:
        """
        Check if the straight-line path from start to goal is blocked by any obstacle.

        Used to guarantee that a detour is required in hard scenarios.
        """
        # Check multiple points along the straight line
        num_checks = 10
        for i in range(num_checks + 1):
            t = i / num_checks
            x = start[0] + t * (goal[0] - start[0])
            y = start[1] + t * (goal[1] - start[1])
            if self.check_collision(np.array([x, y]), robot_radius=0.5):
                return True
        return False

    def _generate_dynamic_obstacles(self, num_obstacles: int):
        """Generate random dynamic obstacles."""
        for _ in range(num_obstacles):
            # Random position (avoiding walls)
            x = self.rng.uniform(1.0, self.width - 1.0)
            y = self.rng.uniform(1.0, self.height - 1.0)

            # Random radius
            radius = self.rng.uniform(0.2, 0.4)

            # Random velocity
            speed = self.rng.uniform(0.2, 0.5)
            angle = self.rng.uniform(0, 2 * np.pi)
            velocity = np.array([speed * np.cos(angle), speed * np.sin(angle)])

            obstacle = DynamicObstacle(
                position=np.array([x, y]),
                radius=radius,
                velocity=velocity,
                bounds=(0.5, self.width - 0.5, 0.5, self.height - 0.5)
            )

            self.dynamic_obstacles.append(obstacle)

    def update_dynamic_obstacles(self, dt: float):
        """Update positions of dynamic obstacles."""
        for obstacle in self.dynamic_obstacles:
            obstacle.update(dt)

    def get_occupancy_grid_at_position(
        self,
        position: np.ndarray,
        robot_radius: float = 0.3
    ) -> np.ndarray:
        """
        Get a local occupancy grid centered at the given position.

        Args:
            position: [x, y] world coordinates
            robot_radius: Robot radius for inflation

        Returns:
            32x32 occupancy grid centered at position
        """
        local_size = 32
        half_size = local_size // 2

        local_grid = np.zeros((local_size, local_size), dtype=np.float32)

        # Convert position to grid coordinates
        center_i, center_j = self._world_to_grid(position[0], position[1])

        # Extract local region from global grid
        for di in range(-half_size, half_size):
            for dj in range(-half_size, half_size):
                i = center_i + di
                j = center_j + dj

                # Local grid indices
                local_i = di + half_size
                local_j = dj + half_size

                # Check bounds and copy value
                if 0 <= i < self.grid_height and 0 <= j < self.grid_width:
                    local_grid[local_i, local_j] = self.occupancy_grid[i, j]
                else:
                    # Outside world bounds = occupied
                    local_grid[local_i, local_j] = 1.0

        return local_grid

    def check_collision(
        self,
        position: np.ndarray,
        robot_radius: float = 0.3,
        ignore_dynamic: bool = False
    ) -> bool:
        """
        Check if a position collides with any obstacle.

        Args:
            position: [x, y] world coordinates
            robot_radius: Robot radius for collision checking
            ignore_dynamic: If True, ignore dynamic obstacles

        Returns:
            True if collision detected
        """
        # Check static obstacles
        for obstacle in self.static_obstacles:
            dist = np.linalg.norm(position - obstacle.position)
            if dist < (robot_radius + obstacle.radius):
                return True

        # Check dynamic obstacles
        if not ignore_dynamic:
            for obstacle in self.dynamic_obstacles:
                dist = np.linalg.norm(position - obstacle.position)
                if dist < (robot_radius + obstacle.radius):
                    return True

        # Check grid boundaries
        if (position[0] < robot_radius or
            position[0] > self.width - robot_radius or
            position[1] < robot_radius or
            position[1] > self.height - robot_radius):
            return True

        return False

    def set_goal(self, position: np.ndarray):
        """Set the goal position."""
        self.goal_position = position.copy()

    def sample_free_position(
        self,
        min_clearance: float = 0.5,
        max_attempts: int = 100
    ) -> Optional[np.ndarray]:
        """
        Sample a random collision-free position.

        Args:
            min_clearance: Minimum distance from obstacles
            max_attempts: Maximum sampling attempts

        Returns:
            [x, y] position or None if failed
        """
        for _ in range(max_attempts):
            x = self.rng.uniform(1.0, self.width - 1.0)
            y = self.rng.uniform(1.0, self.height - 1.0)
            position = np.array([x, y])

            if not self.check_collision(position, min_clearance):
                return position

        return None

    def render(self, robot_position: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Render the grid world as a visualization array.

        Args:
            robot_position: Current robot position for visualization

        Returns:
            RGB image array (H, W, 3)
        """
        # Create RGB image from grid
        h, w = self.grid_height, self.grid_width
        image = np.ones((h, w, 3), dtype=np.float32) * 0.9  # Light gray background

        # Color obstacles
        for i in range(h):
            for j in range(w):
                if self.occupancy_grid[i, j] > 0.5:
                    image[i, j] = [0.3, 0.3, 0.3]  # Dark gray

        # Mark static obstacles in red
        for obstacle in self.static_obstacles:
            i, j = self._world_to_grid(obstacle.position[0], obstacle.position[1])
            if 0 <= i < h and 0 <= j < w:
                radius_cells = int(np.ceil(obstacle.radius / self.grid_resolution))
                for di in range(-radius_cells, radius_cells + 1):
                    for dj in range(-radius_cells, radius_cells + 1):
                        if 0 <= i + di < h and 0 <= j + dj < w:
                            dist = np.sqrt(di**2 + dj**2) * self.grid_resolution
                            if dist <= obstacle.radius:
                                image[i + di, j + dj] = [0.8, 0.2, 0.2]

        # Mark dynamic obstacles in orange
        for obstacle in self.dynamic_obstacles:
            i, j = self._world_to_grid(obstacle.position[0], obstacle.position[1])
            if 0 <= i < h and 0 <= j < w:
                radius_cells = int(np.ceil(obstacle.radius / self.grid_resolution))
                for di in range(-radius_cells, radius_cells + 1):
                    for dj in range(-radius_cells, radius_cells + 1):
                        if 0 <= i + di < h and 0 <= j + dj < w:
                            dist = np.sqrt(di**2 + dj**2) * self.grid_resolution
                            if dist <= obstacle.radius:
                                image[i + di, j + dj] = [1.0, 0.6, 0.0]

        # Mark goal in green
        if self.goal_position is not None:
            i, j = self._world_to_grid(self.goal_position[0], self.goal_position[1])
            if 0 <= i < h and 0 <= j < w:
                goal_radius = 5
                for di in range(-goal_radius, goal_radius + 1):
                    for dj in range(-goal_radius, goal_radius + 1):
                        if 0 <= i + di < h and 0 <= j + dj < w:
                            dist = np.sqrt(di**2 + dj**2)
                            if dist <= goal_radius:
                                image[i + di, j + dj] = [0.2, 0.8, 0.2]

        # Mark robot in blue
        if robot_position is not None:
            i, j = self._world_to_grid(robot_position[0], robot_position[1])
            if 0 <= i < h and 0 <= j < w:
                robot_radius = 3
                for di in range(-robot_radius, robot_radius + 1):
                    for dj in range(-robot_radius, robot_radius + 1):
                        if 0 <= i + di < h and 0 <= j + dj < w:
                            dist = np.sqrt(di**2 + dj**2)
                            if dist <= robot_radius:
                                image[i + di, j + dj] = [0.2, 0.4, 1.0]

        return image

    def reset(self, randomize: bool = True):
        """Reset the grid world, optionally randomizing obstacles."""
        # Clear dynamic obstacles
        for obstacle in self.dynamic_obstacles:
            # Reset to random position
            if randomize:
                obstacle.position = np.array([
                    self.rng.uniform(1.0, self.width - 1.0),
                    self.rng.uniform(1.0, self.height - 1.0)
                ])
                # Random velocity
                speed = self.rng.uniform(0.2, 0.5)
                angle = self.rng.uniform(0, 2 * np.pi)
                obstacle.velocity = np.array([speed * np.cos(angle), speed * np.sin(angle)])
