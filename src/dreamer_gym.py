from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

try:  # optional dependency
    import gymnasium as gym  # type: ignore
    from gymnasium import spaces  # type: ignore
except ImportError:  # lightweight fallback for local testing
    gym = None

    class _FallbackEnv:
        metadata: dict[str, Any] = {}

    class _FallbackBox:
        def __init__(self, low, high, shape, dtype=float):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    class _FallbackSpaces:
        Box = _FallbackBox

    spaces = _FallbackSpaces()

    class _FallbackGym:
        Env = _FallbackEnv

    gym = _FallbackGym()

from src.dreamer_env import (
    DreamerTeamEnvConfig,
    TeamState,
    build_observation,
    compute_reward,
    initial_state,
    step_team_state,
)


def sample_waypoint(*, seed: int | None = None, arena_half_extent: float = 0.68, margin: float = 0.12) -> tuple[float, float]:
    rng = random.Random(seed)
    radius = max(0.0, arena_half_extent - margin)
    return rng.uniform(-radius, radius), rng.uniform(-radius, radius)


@dataclass
class DreamerTeamGymEnv(gym.Env):
    config: DreamerTeamEnvConfig = DreamerTeamEnvConfig()
    max_steps: int = 200
    randomize_waypoint_on_reset: bool = True

    metadata = {"render_modes": ["none"], "name": "DreamerTeamGymEnv-v0"}

    def __post_init__(self) -> None:
        self.observation_space = spaces.Box(low=-1.0e6, high=1.0e6, shape=(19,), dtype=float)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float)
        self._rng = random.Random()
        self.state: TeamState | None = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._rng.seed(seed)
        waypoint = None
        if options and "waypoint" in options:
            waypoint = tuple(options["waypoint"])
        elif self.randomize_waypoint_on_reset:
            waypoint = sample_waypoint(
                seed=self._rng.randint(0, 10**9),
                arena_half_extent=self.config.arena_half_extent,
                margin=0.12,
            )
        self.state = initial_state(waypoint=waypoint or (0.4, 0.0), config=self.config)
        obs = build_observation(self.state)
        return obs, {"waypoint": self.state.waypoint, "state": self.state}

    def step(self, action):
        if self.state is None:
            raise RuntimeError("reset() must be called before step()")
        previous = self.state
        action_tuple = (float(action[0]), float(action[1]))
        current = step_team_state(previous, action=action_tuple, config=self.config)
        reward = float(compute_reward(previous, current))
        self.state = current

        centroid_x = sum(p[0] for p in current.robot_poses) / len(current.robot_poses)
        centroid_y = sum(p[1] for p in current.robot_poses) / len(current.robot_poses)
        dist = ((current.waypoint[0] - centroid_x) ** 2 + (current.waypoint[1] - centroid_y) ** 2) ** 0.5
        terminated = dist < self.config.reach_radius
        truncated = current.step_count >= self.max_steps
        obs = build_observation(current)
        info = {"state": current, "waypoint": current.waypoint, "distance_to_waypoint": dist}
        return obs, reward, terminated, truncated, info

    def render(self):
        return None

    def close(self):
        return None
