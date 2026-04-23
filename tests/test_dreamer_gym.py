import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dreamer_gym import DreamerTeamGymEnv, sample_waypoint  # noqa: E402


class DreamerGymTests(unittest.TestCase):
    def test_sample_waypoint_stays_inside_arena(self):
        for seed in range(5):
            x, y = sample_waypoint(seed=seed, arena_half_extent=0.68, margin=0.12)
            self.assertLessEqual(abs(x), 0.56)
            self.assertLessEqual(abs(y), 0.56)

    def test_reset_returns_observation_and_info(self):
        env = DreamerTeamGymEnv(max_steps=25)
        obs, info = env.reset(seed=7)
        self.assertEqual(len(obs), 19)
        self.assertIn("waypoint", info)
        self.assertIn("state", info)
        self.assertEqual(info["state"].step_count, 0)

    def test_step_returns_gymnasium_style_tuple(self):
        env = DreamerTeamGymEnv(max_steps=5)
        env.reset(seed=3)
        result = env.step((0.3, -0.2))
        self.assertEqual(len(result), 5)
        obs, reward, terminated, truncated, info = result
        self.assertEqual(len(obs), 19)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIn("state", info)

    def test_episode_truncates_at_max_steps(self):
        env = DreamerTeamGymEnv(max_steps=2)
        env.reset(seed=1)
        _, _, _, truncated_1, _ = env.step((0.1, 0.0))
        _, _, _, truncated_2, _ = env.step((0.1, 0.0))
        self.assertFalse(truncated_1)
        self.assertTrue(truncated_2)


if __name__ == "__main__":
    unittest.main()
