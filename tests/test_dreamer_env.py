import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dreamer_env import DreamerTeamEnvConfig, TeamState, build_observation, compute_reward, step_team_state


class DreamerEnvTests(unittest.TestCase):
    def test_build_observation_has_expected_shape(self):
        state = TeamState(
            robot_poses=[
                (-0.06, 0.06, 0.0),
                (0.06, 0.06, 0.0),
                (-0.06, -0.06, 0.0),
                (0.06, -0.06, 0.0),
            ],
            waypoint=(0.3, 0.0),
            team_heading=0.0,
            prev_action=(0.0, 0.0),
            step_count=0,
        )
        obs = build_observation(state)
        self.assertEqual(len(obs), 19)
        self.assertAlmostEqual(obs[0], 0.0, places=6)
        self.assertAlmostEqual(obs[1], 0.0, places=6)

    def test_reward_improves_when_centroid_moves_toward_waypoint(self):
        previous = TeamState(
            robot_poses=[(-0.1, 0.1, 0.0), (0.1, 0.1, 0.0), (-0.1, -0.1, 0.0), (0.1, -0.1, 0.0)],
            waypoint=(0.4, 0.0),
            team_heading=0.0,
            prev_action=(0.0, 0.0),
            step_count=0,
        )
        current = TeamState(
            robot_poses=[(-0.05, 0.1, 0.0), (0.15, 0.1, 0.0), (-0.05, -0.1, 0.0), (0.15, -0.1, 0.0)],
            waypoint=(0.4, 0.0),
            team_heading=0.0,
            prev_action=(0.4, 0.0),
            step_count=1,
        )
        reward = compute_reward(previous, current)
        self.assertGreater(reward, 0.0)

    def test_step_team_state_advances_centroid_forward(self):
        config = DreamerTeamEnvConfig(dt=0.1)
        state = TeamState(
            robot_poses=[
                (-0.06, 0.06, 0.0),
                (0.06, 0.06, 0.0),
                (-0.06, -0.06, 0.0),
                (0.06, -0.06, 0.0),
            ],
            waypoint=(0.4, 0.0),
            team_heading=0.0,
            prev_action=(0.0, 0.0),
            step_count=0,
        )
        next_state = step_team_state(state, action=(0.6, 0.0), config=config)
        before_x = sum(p[0] for p in state.robot_poses) / 4.0
        after_x = sum(p[0] for p in next_state.robot_poses) / 4.0
        self.assertGreater(after_x, before_x)
        self.assertEqual(next_state.step_count, 1)

    def test_rotation_action_changes_team_heading(self):
        config = DreamerTeamEnvConfig(dt=0.1)
        state = TeamState(
            robot_poses=[
                (-0.06, 0.06, 0.0),
                (0.06, 0.06, 0.0),
                (-0.06, -0.06, 0.0),
                (0.06, -0.06, 0.0),
            ],
            waypoint=(0.0, 0.4),
            team_heading=0.0,
            prev_action=(0.0, 0.0),
            step_count=0,
        )
        next_state = step_team_state(state, action=(0.0, 0.8), config=config)
        self.assertNotAlmostEqual(next_state.team_heading, 0.0, places=6)
        self.assertTrue(-math.pi <= next_state.team_heading <= math.pi)


if __name__ == "__main__":
    unittest.main()
