import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dreamer_webots_live import (  # noqa: E402
    DreamerTeamLiveWebotsEnv,
    build_webots_bridge_command,
    snapshot_to_state,
)


class FakeBridge:
    def __init__(self, snapshots):
        self.snapshots = list(snapshots)
        self.reset_calls = []
        self.actions = []
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def send_reset(self, *, waypoint, seed=None):
        self.reset_calls.append({"waypoint": waypoint, "seed": seed})
        return len(self.reset_calls)

    def send_action(self, action):
        self.actions.append(tuple(action))
        return len(self.reset_calls) + len(self.actions)

    def wait_for_snapshot(self, timeout=5.0, min_seq=None):
        if not self.snapshots:
            raise RuntimeError("no snapshots left")
        snapshot = self.snapshots.pop(0)
        if min_seq is not None:
            snapshot = dict(snapshot)
            snapshot.setdefault("seq", min_seq)
        return snapshot

    def close(self):
        self.closed = True


class DreamerWebotsLiveTests(unittest.TestCase):
    def test_snapshot_to_state_extracts_team_state(self):
        snapshot = {
            "seq": 3,
            "step_count": 3,
            "waypoint": [0.4, 0.1],
            "team_heading": 0.2,
            "prev_action": [0.5, -0.1],
            "robots": [
                {"name": "epuck2_A", "pose": [-0.06, 0.06, 0.2]},
                {"name": "epuck2_B", "pose": [0.06, 0.06, 0.2]},
                {"name": "epuck2_C", "pose": [-0.06, -0.06, 0.2]},
                {"name": "epuck2_D", "pose": [0.06, -0.06, 0.2]},
            ],
        }
        state = snapshot_to_state(snapshot)
        self.assertEqual(state.step_count, 3)
        self.assertEqual(len(state.robot_poses), 4)
        self.assertEqual(state.waypoint, (0.4, 0.1))

    def test_reset_uses_bridge_snapshot_and_returns_observation(self):
        env = DreamerTeamLiveWebotsEnv(bridge=FakeBridge([
            {
                "step_count": 0,
                "waypoint": [0.3, 0.0],
                "team_heading": 0.0,
                "prev_action": [0.0, 0.0],
                "robots": [
                    {"name": "epuck2_A", "pose": [-0.06, 0.06, 0.0]},
                    {"name": "epuck2_B", "pose": [0.06, 0.06, 0.0]},
                    {"name": "epuck2_C", "pose": [-0.06, -0.06, 0.0]},
                    {"name": "epuck2_D", "pose": [0.06, -0.06, 0.0]},
                ],
            }
        ]))
        obs, info = env.reset(seed=11)
        self.assertEqual(len(obs), 19)
        self.assertIn("state", info)
        self.assertEqual(info["state"].step_count, 0)
        self.assertEqual(len(env.bridge.reset_calls), 1)

    def test_step_sends_action_and_returns_gym_tuple(self):
        snapshots = [
            {
                "seq": 1,
                "step_count": 0,
                "waypoint": [0.3, 0.0],
                "team_heading": 0.0,
                "prev_action": [0.0, 0.0],
                "robots": [
                    {"name": "epuck2_A", "pose": [-0.06, 0.06, 0.0]},
                    {"name": "epuck2_B", "pose": [0.06, 0.06, 0.0]},
                    {"name": "epuck2_C", "pose": [-0.06, -0.06, 0.0]},
                    {"name": "epuck2_D", "pose": [0.06, -0.06, 0.0]},
                ],
            },
            {
                "seq": 2,
                "step_count": 1,
                "waypoint": [0.3, 0.0],
                "team_heading": 0.05,
                "prev_action": [0.4, 0.1],
                "robots": [
                    {"name": "epuck2_A", "pose": [-0.02, 0.06, 0.05]},
                    {"name": "epuck2_B", "pose": [0.10, 0.06, 0.05]},
                    {"name": "epuck2_C", "pose": [-0.02, -0.06, 0.05]},
                    {"name": "epuck2_D", "pose": [0.10, -0.06, 0.05]},
                ],
            },
        ]
        env = DreamerTeamLiveWebotsEnv(bridge=FakeBridge(snapshots), max_steps=5)
        env.reset(seed=4)
        obs, reward, terminated, truncated, info = env.step((0.4, 0.1))
        self.assertEqual(len(obs), 19)
        self.assertIsInstance(reward, float)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(env.bridge.actions[-1], (0.4, 0.1))
        self.assertIn("distance_to_waypoint", info)

    def test_build_webots_bridge_command_uses_headless_xvfb_without_display(self):
        old_display = os.environ.pop("DISPLAY", None)
        try:
            cmd = build_webots_bridge_command(
                world_path=ROOT / "worlds/epuck2_dreamer_team_4.wbt",
                bridge_dir="/tmp/dreamer_team_bridge",
                mode="auto",
            )
        finally:
            if old_display is not None:
                os.environ["DISPLAY"] = old_display
        self.assertTrue(cmd[0].endswith("scripts/launch_webots_headless.sh"))
        self.assertTrue(cmd[1].endswith("worlds/epuck2_dreamer_team_4.wbt"))
        self.assertIn("--no-rendering", cmd)
        self.assertIn("--mode=realtime", cmd)
        self.assertTrue(any(arg.startswith("--port=") for arg in cmd))

    def test_build_webots_bridge_command_uses_gui_when_display_exists(self):
        old_display = os.environ.get("DISPLAY")
        os.environ["DISPLAY"] = ":0"
        try:
            cmd = build_webots_bridge_command(
                world_path=ROOT / "worlds/epuck2_dreamer_team_4.wbt",
                bridge_dir="/tmp/dreamer_team_bridge",
                mode="auto",
            )
        finally:
            if old_display is None:
                os.environ.pop("DISPLAY", None)
            else:
                os.environ["DISPLAY"] = old_display
        self.assertEqual(cmd[0], "webots")
        self.assertIn("--batch", cmd)
        self.assertIn("--no-rendering", cmd)
        self.assertTrue(any(arg.startswith("--port=") for arg in cmd))


if __name__ == "__main__":
    unittest.main()
