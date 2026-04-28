"""Shared logic for the four-e-puck virtual-body controller.

This module keeps the core kinematics pure so it can be unit-tested
outside Webots. The runtime supervisors/controllers import these helpers
and add device access on top.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Iterable

ROLE_ORDER = ["front_left", "front_right", "rear_left", "rear_right"]
ROLE_SLOT_SIGNS = {
    "front_left": (1.0, 1.0),
    "front_right": (1.0, -1.0),
    "rear_left": (-1.0, 1.0),
    "rear_right": (-1.0, -1.0),
}


@dataclass(frozen=True)
class ObstacleBox:
    center: tuple[float, float]
    size: tuple[float, float]
    yaw: float


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def normalize_angle(angle: float) -> float:
    wrapped = (angle + math.pi) % (2.0 * math.pi) - math.pi
    return -math.pi if math.isclose(wrapped, math.pi) else wrapped


def compute_centroid(poses: Iterable[tuple[float, float, float]]) -> tuple[float, float]:
    poses = list(poses)
    if not poses:
        raise ValueError("poses cannot be empty")
    x = sum(p[0] for p in poses) / len(poses)
    y = sum(p[1] for p in poses) / len(poses)
    return x, y


def point_in_obstacle(
    point: tuple[float, float],
    obstacle: ObstacleBox,
    *,
    clearance: float,
) -> bool:
    px = point[0] - obstacle.center[0]
    py = point[1] - obstacle.center[1]
    c = math.cos(-obstacle.yaw)
    s = math.sin(-obstacle.yaw)
    local_x = px * c - py * s
    local_y = px * s + py * c
    half_x = obstacle.size[0] / 2.0 + clearance
    half_y = obstacle.size[1] / 2.0 + clearance
    return abs(local_x) <= half_x and abs(local_y) <= half_y


def point_is_navigable(
    point: tuple[float, float],
    *,
    obstacles: list[ObstacleBox],
    arena_half_extent: float,
    formation_radius: float,
) -> bool:
    limit = arena_half_extent - formation_radius
    if abs(point[0]) > limit or abs(point[1]) > limit:
        return False
    return not any(point_in_obstacle(point, obstacle, clearance=formation_radius) for obstacle in obstacles)


def segment_is_clear(
    start: tuple[float, float],
    goal: tuple[float, float],
    *,
    obstacles: list[ObstacleBox],
    arena_half_extent: float,
    formation_radius: float,
    sample_step: float = 0.03,
) -> bool:
    distance = math.hypot(goal[0] - start[0], goal[1] - start[1])
    samples = max(2, int(math.ceil(distance / sample_step)) + 1)
    for index in range(samples):
        t = index / (samples - 1)
        point = (
            start[0] + (goal[0] - start[0]) * t,
            start[1] + (goal[1] - start[1]) * t,
        )
        if not point_is_navigable(
            point,
            obstacles=obstacles,
            arena_half_extent=arena_half_extent,
            formation_radius=formation_radius,
        ):
            return False
    return True


def _grid_neighbors(node: tuple[int, int]) -> list[tuple[int, int]]:
    x, y = node
    return [
        (x + dx, y + dy)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if not (dx == 0 and dy == 0)
    ]


def _node_to_point(node: tuple[int, int], *, arena_half_extent: float, grid_resolution: float) -> tuple[float, float]:
    return (
        -arena_half_extent + node[0] * grid_resolution,
        -arena_half_extent + node[1] * grid_resolution,
    )


def _point_to_node(point: tuple[float, float], *, arena_half_extent: float, grid_resolution: float) -> tuple[int, int]:
    return (
        int(round((point[0] + arena_half_extent) / grid_resolution)),
        int(round((point[1] + arena_half_extent) / grid_resolution)),
    )


def _smooth_path(
    path: list[tuple[float, float]],
    *,
    obstacles: list[ObstacleBox],
    arena_half_extent: float,
    formation_radius: float,
) -> list[tuple[float, float]]:
    if len(path) <= 2:
        return path
    smoothed = [path[0]]
    index = 0
    while index < len(path) - 1:
        look_ahead = len(path) - 1
        while look_ahead > index + 1:
            if segment_is_clear(
                path[index],
                path[look_ahead],
                obstacles=obstacles,
                arena_half_extent=arena_half_extent,
                formation_radius=formation_radius,
            ):
                break
            look_ahead -= 1
        smoothed.append(path[look_ahead])
        index = look_ahead
    return smoothed


def plan_team_path(
    *,
    start: tuple[float, float],
    goal: tuple[float, float],
    obstacles: list[ObstacleBox],
    arena_half_extent: float,
    formation_radius: float,
    grid_resolution: float = 0.05,
) -> list[tuple[float, float]]:
    if segment_is_clear(
        start,
        goal,
        obstacles=obstacles,
        arena_half_extent=arena_half_extent,
        formation_radius=formation_radius,
    ):
        return [start, goal]

    max_index = int(round((2.0 * arena_half_extent) / grid_resolution))
    start_node = _point_to_node(start, arena_half_extent=arena_half_extent, grid_resolution=grid_resolution)
    goal_node = _point_to_node(goal, arena_half_extent=arena_half_extent, grid_resolution=grid_resolution)

    def node_valid(node: tuple[int, int]) -> bool:
        if node[0] < 0 or node[1] < 0 or node[0] > max_index or node[1] > max_index:
            return False
        point = _node_to_point(node, arena_half_extent=arena_half_extent, grid_resolution=grid_resolution)
        return point_is_navigable(
            point,
            obstacles=obstacles,
            arena_half_extent=arena_half_extent,
            formation_radius=formation_radius,
        )

    frontier: list[tuple[float, tuple[int, int]]] = []
    heapq.heappush(frontier, (0.0, start_node))
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start_node: None}
    cost_so_far: dict[tuple[int, int], float] = {start_node: 0.0}

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal_node:
            break
        for neighbor in _grid_neighbors(current):
            if not node_valid(neighbor):
                continue
            step_cost = math.hypot(neighbor[0] - current[0], neighbor[1] - current[1])
            new_cost = cost_so_far[current] + step_cost
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                heuristic = math.hypot(goal_node[0] - neighbor[0], goal_node[1] - neighbor[1])
                heapq.heappush(frontier, (new_cost + heuristic, neighbor))
                came_from[neighbor] = current

    if goal_node not in came_from:
        return [start, goal]

    nodes: list[tuple[int, int]] = []
    current = goal_node
    while current is not None:
        nodes.append(current)
        current = came_from[current]
    nodes.reverse()

    path = [start]
    for node in nodes[1:-1]:
        path.append(_node_to_point(node, arena_half_extent=arena_half_extent, grid_resolution=grid_resolution))
    path.append(goal)
    return _smooth_path(
        path,
        obstacles=obstacles,
        arena_half_extent=arena_half_extent,
        formation_radius=formation_radius,
    )


def role_target_position(
    role: str,
    *,
    centroid: tuple[float, float],
    team_heading: float,
    spacing: float,
) -> tuple[float, float]:
    if role not in ROLE_SLOT_SIGNS:
        raise KeyError(f"unknown role: {role}")
    longitudinal_sign, lateral_sign = ROLE_SLOT_SIGNS[role]
    forward = longitudinal_sign * spacing
    lateral = lateral_sign * spacing
    c = math.cos(team_heading)
    s = math.sin(team_heading)
    offset_x = forward * c - lateral * s
    offset_y = forward * s + lateral * c
    return centroid[0] + offset_x, centroid[1] + offset_y


def tracking_command(
    *,
    pose: tuple[float, float, float],
    target: tuple[float, float],
    max_speed: float,
    cruise_gain: float = 0.82,
    turn_gain: float = 2.4,
    distance_gain: float = 11.0,
) -> tuple[float, float]:
    x, y, heading = pose
    dx = target[0] - x
    dy = target[1] - y
    distance = math.hypot(dx, dy)
    desired_heading = math.atan2(dy, dx)
    heading_error = normalize_angle(desired_heading - heading)

    forward = clamp(distance * distance_gain, -cruise_gain * max_speed, cruise_gain * max_speed)
    turn = clamp(heading_error * turn_gain, -0.75 * max_speed, 0.75 * max_speed)

    if abs(heading_error) > 1.35:
        forward *= 0.15

    left = clamp(forward - turn, -max_speed, max_speed)
    right = clamp(forward + turn, -max_speed, max_speed)
    return left, right


def blend_obstacle_avoidance(
    *,
    desired_left: float,
    desired_right: float,
    sensor_values: list[float],
    max_speed: float,
) -> tuple[float, float]:
    normalized = [min(1.0, max(0.0, value / 4096.0)) for value in sensor_values]
    front_right = max(normalized[0], normalized[1])
    front_left = max(normalized[6], normalized[7])
    side_right = normalized[2]
    side_left = normalized[5]
    crowd = max(front_right, front_left, side_right, side_left)

    left = desired_left
    right = desired_right
    if crowd > 0.18:
        slowdown = clamp(1.0 - (0.85 * crowd), 0.15, 1.0)
        left *= slowdown
        right *= slowdown

        if front_left > front_right:
            left += 0.16 * max_speed
            right -= 0.28 * max_speed * front_left
        elif front_right > front_left:
            left -= 0.28 * max_speed * front_right
            right += 0.16 * max_speed

        left += 0.08 * max_speed * side_left
        right -= 0.08 * max_speed * side_right

    return clamp(left, -max_speed, max_speed), clamp(right, -max_speed, max_speed)
