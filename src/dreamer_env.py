from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from controllers.dreamer_team_common import compute_centroid, normalize_angle, role_target_position, ROLE_ORDER


@dataclass(frozen=True)
class DreamerTeamEnvConfig:
    slot_spacing: float = 0.06
    dt: float = 0.1
    max_forward_speed: float = 0.22
    max_turn_rate: float = 1.4
    arena_half_extent: float = 0.68
    reach_radius: float = 0.08


@dataclass(frozen=True)
class TeamState:
    robot_poses: list[tuple[float, float, float]]
    waypoint: tuple[float, float]
    team_heading: float
    prev_action: tuple[float, float]
    step_count: int


def _team_centroid(state: TeamState) -> tuple[float, float]:
    return compute_centroid(state.robot_poses)


def _clip_position(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def _formation_poses(
    *,
    centroid: tuple[float, float],
    team_heading: float,
    slot_spacing: float,
) -> list[tuple[float, float, float]]:
    poses = []
    for role in ROLE_ORDER:
        x, y = role_target_position(role, centroid=centroid, team_heading=team_heading, spacing=slot_spacing)
        poses.append((x, y, team_heading))
    return poses


def build_observation(state: TeamState) -> list[float]:
    centroid = _team_centroid(state)
    waypoint_dx = state.waypoint[0] - centroid[0]
    waypoint_dy = state.waypoint[1] - centroid[1]
    obs = [centroid[0], centroid[1], waypoint_dx, waypoint_dy, state.team_heading, *state.prev_action]
    for x, y, heading in state.robot_poses:
        obs.extend([x - centroid[0], y - centroid[1], normalize_angle(heading - state.team_heading)])
    return obs


def step_team_state(
    state: TeamState,
    *,
    action: tuple[float, float],
    config: DreamerTeamEnvConfig,
) -> TeamState:
    forward_cmd = max(-1.0, min(1.0, float(action[0])))
    turn_cmd = max(-1.0, min(1.0, float(action[1])))
    forward_speed = forward_cmd * config.max_forward_speed
    turn_rate = turn_cmd * config.max_turn_rate

    new_heading = normalize_angle(state.team_heading + turn_rate * config.dt)
    centroid = _team_centroid(state)
    new_centroid = (
        _clip_position(centroid[0] + math.cos(new_heading) * forward_speed * config.dt, config.arena_half_extent),
        _clip_position(centroid[1] + math.sin(new_heading) * forward_speed * config.dt, config.arena_half_extent),
    )
    return TeamState(
        robot_poses=_formation_poses(centroid=new_centroid, team_heading=new_heading, slot_spacing=config.slot_spacing),
        waypoint=state.waypoint,
        team_heading=new_heading,
        prev_action=(forward_cmd, turn_cmd),
        step_count=state.step_count + 1,
    )


def compute_reward(previous: TeamState, current: TeamState, *, alive_bonus: float = 0.01) -> float:
    prev_centroid = _team_centroid(previous)
    cur_centroid = _team_centroid(current)
    prev_dist = math.dist(prev_centroid, previous.waypoint)
    cur_dist = math.dist(cur_centroid, current.waypoint)
    progress = prev_dist - cur_dist
    heading_bonus = 0.02 * math.cos(normalize_angle(math.atan2(current.waypoint[1] - cur_centroid[1], current.waypoint[0] - cur_centroid[0]) - current.team_heading))
    reach_bonus = 1.0 if cur_dist < 0.08 else 0.0
    action_penalty = 0.01 * (abs(current.prev_action[0]) + abs(current.prev_action[1]))
    return progress * 4.0 + heading_bonus + reach_bonus + alive_bonus - action_penalty


def initial_state(
    *,
    waypoint: tuple[float, float] = (0.4, 0.0),
    config: DreamerTeamEnvConfig | None = None,
) -> TeamState:
    config = config or DreamerTeamEnvConfig()
    centroid = (0.0, 0.0)
    team_heading = 0.0
    return TeamState(
        robot_poses=_formation_poses(centroid=centroid, team_heading=team_heading, slot_spacing=config.slot_spacing),
        waypoint=waypoint,
        team_heading=team_heading,
        prev_action=(0.0, 0.0),
        step_count=0,
    )
