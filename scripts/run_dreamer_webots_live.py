#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dreamer_webots_live import BridgePaths, DreamerTeamLiveWebotsEnv, FileBridgeClient  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a short live Webots rollout through the Dreamer bridge.")
    parser.add_argument(
        "--world",
        type=Path,
        default=ROOT / "worlds/epuck2_dreamer_team_4.wbt",
        help="Path to the Webots world file.",
    )
    parser.add_argument("--steps", type=int, default=60, help="Maximum number of env steps to run.")
    parser.add_argument(
        "--bridge-dir",
        type=Path,
        default=None,
        help="Optional bridge directory. Defaults to a fresh temporary directory.",
    )
    parser.add_argument(
        "--launch-mode",
        choices=["auto", "gui", "headless"],
        default="auto",
        help="How to launch Webots when the env starts it.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Reset seed.")
    parser.add_argument(
        "--snapshot-timeout",
        type=float,
        default=20.0,
        help="Seconds to wait for each snapshot from Webots.",
    )
    parser.add_argument(
        "--keep-bridge-dir",
        action="store_true",
        help="Keep the bridge directory after the script exits.",
    )
    return parser.parse_args()


def scripted_action(step: int) -> tuple[float, float]:
    if step < 15:
        return (0.6, 0.0)
    if step < 30:
        return (0.6, 0.25)
    if step < 45:
        return (0.45, -0.2)
    return (0.35, 0.0)


def main() -> int:
    args = parse_args()
    temp_bridge_dir = False
    bridge_dir = args.bridge_dir
    if bridge_dir is None:
        bridge_dir = Path(tempfile.mkdtemp(prefix="dreamer-team-bridge-"))
        temp_bridge_dir = True

    bridge = FileBridgeClient(
        paths=BridgePaths.from_dir(bridge_dir),
        world_path=args.world,
        mode=args.launch_mode,
    )
    env = DreamerTeamLiveWebotsEnv(bridge=bridge, max_steps=args.steps, snapshot_timeout=args.snapshot_timeout)

    try:
        obs, info = env.reset(seed=args.seed)
        rollout = [
            {
                "type": "reset",
                "obs_dim": len(obs),
                "waypoint": info["waypoint"],
                "bridge_dir": str(bridge_dir),
            }
        ]
        for step in range(args.steps):
            action = scripted_action(step)
            obs, reward, terminated, truncated, info = env.step(action)
            rollout.append(
                {
                    "type": "step",
                    "step": step + 1,
                    "action": list(action),
                    "reward": round(float(reward), 5),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "distance_to_waypoint": round(float(info["distance_to_waypoint"]), 5),
                    "team_heading": round(float(info["state"].team_heading), 5),
                    "centroid": [
                        round(sum(p[0] for p in info["state"].robot_poses) / len(info["state"].robot_poses), 5),
                        round(sum(p[1] for p in info["state"].robot_poses) / len(info["state"].robot_poses), 5),
                    ],
                    "obs_dim": len(obs),
                    "seq": info["snapshot"].get("seq"),
                }
            )
            if terminated or truncated:
                break
        print(json.dumps(rollout, indent=2))
        return 0
    finally:
        env.close()
        if temp_bridge_dir and args.keep_bridge_dir:
            print(f"Kept bridge directory: {bridge_dir}", file=sys.stderr)
        elif temp_bridge_dir or not args.keep_bridge_dir:
            shutil.rmtree(bridge_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
