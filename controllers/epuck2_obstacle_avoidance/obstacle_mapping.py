from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw

SENSOR_ANGLES = [0.35, 0.9, 1.57, 2.7, -2.7, -1.57, -0.9, -0.35]
SENSOR_DETECTION_THRESHOLD = 0.08
MIN_SENSOR_RANGE_M = 0.035
MAX_SENSOR_RANGE_M = 0.16
WHEEL_RADIUS_M = 0.0205
AXLE_LENGTH_M = 0.053
ROBOT_MARK_RADIUS_CELLS = 2


@dataclass
class Pose2D:
    x: float
    y: float
    theta: float


@dataclass
class MappingCompletionTracker:
    min_known_cells: int = 220
    stable_growth_threshold: int = 2
    stable_updates_required: int = 6
    previous_known_cells: int = 0
    stable_updates: int = 0

    def update(self, state: "MappingState") -> bool:
        known_cells = count_known_cells(state)
        growth = max(0, known_cells - self.previous_known_cells)

        if known_cells >= self.min_known_cells and growth <= self.stable_growth_threshold:
            self.stable_updates += 1
        else:
            self.stable_updates = 0

        self.previous_known_cells = known_cells
        return self.stable_updates >= self.stable_updates_required


@dataclass
class MappingState:
    width: int = 160
    height: int = 160
    meters_per_cell: float = 0.01
    unknown_fill: int = 127
    free_decrement: int = 10
    occupied_increment: int = 18
    grid: list[list[int]] = field(init=False)
    origin_x: int = field(init=False)
    origin_y: int = field(init=False)

    def __post_init__(self):
        self.UNKNOWN = self.unknown_fill
        self.grid = [
            [self.unknown_fill for _ in range(self.width)]
            for _ in range(self.height)
        ]
        self.origin_x = self.width // 2
        self.origin_y = self.height // 2

    def world_to_cell(self, x_m: float, y_m: float) -> tuple[int, int]:
        cell_x = self.origin_x + int(round(x_m / self.meters_per_cell))
        cell_y = self.origin_y - int(round(y_m / self.meters_per_cell))
        return cell_x, cell_y

    def in_bounds(self, cell_x: int, cell_y: int) -> bool:
        return 0 <= cell_x < self.width and 0 <= cell_y < self.height

    def mark_free(self, cell_x: int, cell_y: int):
        if self.in_bounds(cell_x, cell_y):
            self.grid[cell_y][cell_x] = max(0, self.grid[cell_y][cell_x] - self.free_decrement)

    def mark_occupied(self, cell_x: int, cell_y: int):
        if self.in_bounds(cell_x, cell_y):
            self.grid[cell_y][cell_x] = min(255, self.grid[cell_y][cell_x] + self.occupied_increment)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def count_known_cells(state: MappingState) -> int:
    return sum(
        1
        for row in state.grid
        for value in row
        if value != state.UNKNOWN
    )


def normalize_angle(theta: float) -> float:
    return math.atan2(math.sin(theta), math.cos(theta))


def sensor_hit_distance(normalized_sensor_value: float) -> float | None:
    if normalized_sensor_value < SENSOR_DETECTION_THRESHOLD:
        return None

    clipped = clamp(normalized_sensor_value, 0.0, 1.0)
    span = MAX_SENSOR_RANGE_M - MIN_SENSOR_RANGE_M
    return MIN_SENSOR_RANGE_M + (1.0 - clipped) * span


def update_pose(
    pose: Pose2D,
    left_speed: float,
    right_speed: float,
    dt: float,
    wheel_radius: float = WHEEL_RADIUS_M,
    axle_length: float = AXLE_LENGTH_M,
) -> Pose2D:
    linear_velocity = 0.5 * wheel_radius * (left_speed + right_speed)
    angular_velocity = wheel_radius * (right_speed - left_speed) / axle_length
    next_theta = normalize_angle(pose.theta + angular_velocity * dt)
    heading = pose.theta + 0.5 * angular_velocity * dt
    next_x = pose.x + linear_velocity * math.cos(heading) * dt
    next_y = pose.y + linear_velocity * math.sin(heading) * dt
    return Pose2D(next_x, next_y, next_theta)


def _mark_line_free(state: MappingState, start: tuple[int, int], end: tuple[int, int]):
    x0, y0 = start
    x1, y1 = end
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    for i in range(steps):
        alpha = i / steps
        cell_x = int(round(x0 + alpha * (x1 - x0)))
        cell_y = int(round(y0 + alpha * (y1 - y0)))
        state.mark_free(cell_x, cell_y)


def mark_sensor_observations(
    state: MappingState,
    pose: Pose2D,
    normalized_sensors: list[float],
    sensor_angles: list[float] = SENSOR_ANGLES,
):
    robot_cell = state.world_to_cell(pose.x, pose.y)

    for dx in range(-ROBOT_MARK_RADIUS_CELLS, ROBOT_MARK_RADIUS_CELLS + 1):
        for dy in range(-ROBOT_MARK_RADIUS_CELLS, ROBOT_MARK_RADIUS_CELLS + 1):
            state.mark_free(robot_cell[0] + dx, robot_cell[1] + dy)

    for sensor_value, sensor_angle in zip(normalized_sensors, sensor_angles):
        hit_distance = sensor_hit_distance(sensor_value)
        if hit_distance is None:
            continue

        ray_angle = pose.theta + sensor_angle
        hit_x = pose.x + hit_distance * math.cos(ray_angle)
        hit_y = pose.y + hit_distance * math.sin(ray_angle)
        hit_cell = state.world_to_cell(hit_x, hit_y)
        _mark_line_free(state, robot_cell, hit_cell)
        state.mark_occupied(*hit_cell)


def save_map_frame(
    state: MappingState,
    output_dir: str | Path,
    frame_index: int,
    pose: Pose2D | None = None,
    scale: int = 4,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_path = output_dir / f"frame_{frame_index:04d}.png"
    save_map_image(state, frame_path, pose=pose, scale=scale)
    return frame_path


def save_map_image(state: MappingState, output_path: str | Path, pose: Pose2D | None = None, scale: int = 4):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new("L", (state.width, state.height))
    pixels = image.load()
    for y, row in enumerate(state.grid):
        for x, value in enumerate(row):
            pixels[x, y] = value

    if pose is not None:
        current_pose = pose
        draw = ImageDraw.Draw(image)
        cell_x, cell_y = state.world_to_cell(current_pose.x, current_pose.y)
        draw.ellipse(
            (
                cell_x - 2,
                cell_y - 2,
                cell_x + 2,
                cell_y + 2,
            ),
            fill=220,
        )

    if scale > 1:
        image = image.resize((state.width * scale, state.height * scale), resample=Image.Resampling.NEAREST)

    image.save(output_path)
