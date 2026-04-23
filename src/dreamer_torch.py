from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch

from src.dreamer_env import DreamerTeamEnvConfig, TeamState

ROLE_SIGNS = torch.tensor(
    [
        [1.0, 1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
        [-1.0, -1.0],
    ],
    dtype=torch.float32,
)


def normalize_angle_tensor(angle: torch.Tensor) -> torch.Tensor:
    wrapped = torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi
    return torch.where(torch.isclose(wrapped, torch.full_like(wrapped, math.pi)), -math.pi, wrapped)


@dataclass(frozen=True)
class TorchPolicySpec:
    kind: str
    obs_dim: int
    action_dim: int
    hidden_sizes: tuple[int, ...] = ()

    @property
    def layer_sizes(self) -> tuple[int, ...]:
        return (self.obs_dim, *self.hidden_sizes, self.action_dim)

    @property
    def param_dim(self) -> int:
        total = 0
        for in_dim, out_dim in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            total += in_dim * out_dim + out_dim
        return total


@dataclass
class TorchMLPPolicy:
    weights: list[torch.Tensor]
    biases: list[torch.Tensor]

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs
        for idx, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            x = x @ weight.T + bias
            x = torch.tanh(x)
            if idx == len(self.weights) - 1:
                x = torch.clamp(x, -1.0, 1.0)
        return x


@dataclass
class TorchTeamState:
    centroid: torch.Tensor
    waypoint: torch.Tensor
    team_heading: torch.Tensor
    prev_action: torch.Tensor
    step_count: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.centroid.shape[0])

    def robot_poses(self, config: DreamerTeamEnvConfig) -> torch.Tensor:
        return formation_poses_batch(self.centroid, self.team_heading, slot_spacing=config.slot_spacing)


def policy_from_flat(vector: torch.Tensor, policy_spec: TorchPolicySpec, *, device: str | torch.device = "cpu") -> TorchMLPPolicy:
    vector = torch.as_tensor(vector, dtype=torch.float32, device=device).reshape(-1)
    if vector.shape[0] != policy_spec.param_dim:
        raise ValueError(f"policy vector has wrong shape {tuple(vector.shape)}, expected {(policy_spec.param_dim,)}")
    weights: list[torch.Tensor] = []
    biases: list[torch.Tensor] = []
    offset = 0
    for in_dim, out_dim in zip(policy_spec.layer_sizes[:-1], policy_spec.layer_sizes[1:]):
        weight_count = in_dim * out_dim
        weights.append(vector[offset: offset + weight_count].reshape(out_dim, in_dim))
        offset += weight_count
        biases.append(vector[offset: offset + out_dim])
        offset += out_dim
    return TorchMLPPolicy(weights=weights, biases=biases)


def state_from_team_states(states: Iterable[TeamState], *, device: str | torch.device = "cpu") -> TorchTeamState:
    states = list(states)
    centroids = []
    waypoints = []
    headings = []
    prev_actions = []
    step_counts = []
    for state in states:
        xs = [pose[0] for pose in state.robot_poses]
        ys = [pose[1] for pose in state.robot_poses]
        centroids.append((sum(xs) / len(xs), sum(ys) / len(ys)))
        waypoints.append(state.waypoint)
        headings.append(state.team_heading)
        prev_actions.append(state.prev_action)
        step_counts.append(state.step_count)
    return TorchTeamState(
        centroid=torch.tensor(centroids, dtype=torch.float32, device=device),
        waypoint=torch.tensor(waypoints, dtype=torch.float32, device=device),
        team_heading=torch.tensor(headings, dtype=torch.float32, device=device),
        prev_action=torch.tensor(prev_actions, dtype=torch.float32, device=device),
        step_count=torch.tensor(step_counts, dtype=torch.int64, device=device),
    )


def formation_poses_batch(centroid: torch.Tensor, team_heading: torch.Tensor, *, slot_spacing: float) -> torch.Tensor:
    role_signs = ROLE_SIGNS.to(device=centroid.device, dtype=centroid.dtype)
    forward = role_signs[:, 0] * slot_spacing
    lateral = role_signs[:, 1] * slot_spacing
    c = torch.cos(team_heading).unsqueeze(1)
    s = torch.sin(team_heading).unsqueeze(1)
    offset_x = forward.unsqueeze(0) * c - lateral.unsqueeze(0) * s
    offset_y = forward.unsqueeze(0) * s + lateral.unsqueeze(0) * c
    x = centroid[:, 0:1] + offset_x
    y = centroid[:, 1:2] + offset_y
    heading = team_heading.unsqueeze(1).expand(-1, 4)
    return torch.stack([x, y, heading], dim=-1)


def build_observation_batch(state: TorchTeamState, *, config: DreamerTeamEnvConfig | None = None) -> torch.Tensor:
    config = config or DreamerTeamEnvConfig()
    robot_poses = state.robot_poses(config)
    centroid = state.centroid
    waypoint_delta = state.waypoint - centroid
    pieces = [centroid, waypoint_delta, state.team_heading.unsqueeze(1), state.prev_action]
    rel_xy = robot_poses[:, :, 0:2] - centroid.unsqueeze(1)
    rel_heading = normalize_angle_tensor(robot_poses[:, :, 2] - state.team_heading.unsqueeze(1)).unsqueeze(-1)
    robot_features = torch.cat([rel_xy, rel_heading], dim=-1).reshape(state.batch_size, -1)
    pieces.append(robot_features)
    return torch.cat(pieces, dim=1)


def step_batch(state: TorchTeamState, action: torch.Tensor, *, config: DreamerTeamEnvConfig) -> TorchTeamState:
    action = torch.as_tensor(action, dtype=state.centroid.dtype, device=state.centroid.device)
    forward_cmd = torch.clamp(action[:, 0], -1.0, 1.0)
    turn_cmd = torch.clamp(action[:, 1], -1.0, 1.0)
    forward_speed = forward_cmd * config.max_forward_speed
    turn_rate = turn_cmd * config.max_turn_rate
    new_heading = normalize_angle_tensor(state.team_heading + turn_rate * config.dt)
    delta = torch.stack([torch.cos(new_heading), torch.sin(new_heading)], dim=1) * (forward_speed * config.dt).unsqueeze(1)
    new_centroid = torch.clamp(state.centroid + delta, -config.arena_half_extent, config.arena_half_extent)
    return TorchTeamState(
        centroid=new_centroid,
        waypoint=state.waypoint,
        team_heading=new_heading,
        prev_action=torch.stack([forward_cmd, turn_cmd], dim=1),
        step_count=state.step_count + 1,
    )


def compute_reward_batch(previous: TorchTeamState, current: TorchTeamState, *, alive_bonus: float = 0.01) -> torch.Tensor:
    prev_dist = torch.linalg.norm(previous.waypoint - previous.centroid, dim=1)
    cur_dist = torch.linalg.norm(current.waypoint - current.centroid, dim=1)
    progress = prev_dist - cur_dist
    waypoint_angle = torch.atan2(current.waypoint[:, 1] - current.centroid[:, 1], current.waypoint[:, 0] - current.centroid[:, 0])
    heading_error = normalize_angle_tensor(waypoint_angle - current.team_heading)
    heading_bonus = 0.02 * torch.cos(heading_error)
    reach_bonus = torch.where(cur_dist < 0.08, torch.ones_like(cur_dist), torch.zeros_like(cur_dist))
    action_penalty = 0.01 * torch.abs(current.prev_action).sum(dim=1)
    return progress * 4.0 + heading_bonus + reach_bonus + alive_bonus - action_penalty
