"""Shared logic for the four-e-puck virtual-body controller.

This module keeps the core kinematics pure so it can be unit-tested
outside Webots. The runtime supervisors/controllers import these helpers
and add device access on top.
"""

from __future__ import annotations

import math
from typing import Iterable

ROLE_ORDER = ["front_left", "front_right", "rear_left", "rear_right"]
ROLE_SLOT_SIGNS = {
    "front_left": (1.0, 1.0),
    "front_right": (1.0, -1.0),
    "rear_left": (-1.0, 1.0),
    "rear_right": (-1.0, -1.0),
}


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
