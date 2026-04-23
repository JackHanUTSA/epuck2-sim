#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dreamer_gym import DreamerTeamGymEnv  # noqa: E402


def main() -> int:
    env = DreamerTeamGymEnv(max_steps=8)
    obs, info = env.reset(seed=42)
    rollout = [{"type": "reset", "obs_dim": len(obs), "waypoint": info["waypoint"]}]
    for action in [(0.5, 0.0), (0.5, 0.2), (0.3, -0.3)]:
        obs, reward, terminated, truncated, info = env.step(action)
        rollout.append(
            {
                "type": "step",
                "action": action,
                "reward": round(reward, 4),
                "terminated": terminated,
                "truncated": truncated,
                "distance_to_waypoint": round(info["distance_to_waypoint"], 4),
                "obs_dim": len(obs),
            }
        )
        if terminated or truncated:
            break
    print(json.dumps(rollout, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
