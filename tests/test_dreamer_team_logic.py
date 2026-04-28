import math
import sys
import unittest
from pathlib import Path

CONTROLLERS_DIR = Path(__file__).resolve().parents[1] / "controllers"
if str(CONTROLLERS_DIR) not in sys.path:
    sys.path.insert(0, str(CONTROLLERS_DIR))

from dreamer_team_common import (  # noqa: E402
    ROLE_ORDER,
    ObstacleBox,
    compute_centroid,
    normalize_angle,
    plan_team_path,
    role_target_position,
    tracking_command,
)


class DreamerTeamLogicTests(unittest.TestCase):
    def test_compute_centroid_of_four_robot_block(self):
        poses = [
            (0.1, 0.1, 0.0),
            (-0.1, 0.1, 0.0),
            (-0.1, -0.1, 0.0),
            (0.1, -0.1, 0.0),
        ]
        cx, cy = compute_centroid(poses)
        self.assertAlmostEqual(cx, 0.0, places=6)
        self.assertAlmostEqual(cy, 0.0, places=6)

    def test_role_target_position_rotates_with_team_heading(self):
        x, y = role_target_position("front_left", centroid=(0.0, 0.0), team_heading=math.pi / 2, spacing=0.1)
        self.assertAlmostEqual(x, -0.1, places=5)
        self.assertAlmostEqual(y, 0.1, places=5)

    def test_normalize_angle_wraps_to_minus_pi_pi(self):
        wrapped = normalize_angle(3 * math.pi)
        self.assertAlmostEqual(wrapped, -math.pi, places=6)

    def test_tracking_command_turns_robot_toward_target(self):
        left, right = tracking_command(
            pose=(0.0, 0.0, 0.0),
            target=(0.0, 1.0),
            max_speed=6.28,
        )
        self.assertLess(left, right)
        self.assertLessEqual(abs(left), 6.28)
        self.assertLessEqual(abs(right), 6.28)

    def test_role_order_is_four_robot_virtual_body(self):
        self.assertEqual(ROLE_ORDER, ["front_left", "front_right", "rear_left", "rear_right"])

    def test_plan_team_path_returns_direct_segment_when_clear(self):
        path = plan_team_path(
            start=(-0.5, -0.5),
            goal=(0.5, 0.5),
            obstacles=[],
            arena_half_extent=0.7,
            formation_radius=0.12,
            grid_resolution=0.05,
        )
        self.assertEqual(path[0], (-0.5, -0.5))
        self.assertEqual(path[-1], (0.5, 0.5))
        self.assertEqual(len(path), 2)

    def test_plan_team_path_detours_around_center_obstacle(self):
        path = plan_team_path(
            start=(-0.5, 0.0),
            goal=(0.5, 0.0),
            obstacles=[ObstacleBox(center=(0.0, 0.0), size=(0.20, 0.20), yaw=0.0)],
            arena_half_extent=0.7,
            formation_radius=0.12,
            grid_resolution=0.05,
        )
        self.assertEqual(path[0], (-0.5, 0.0))
        self.assertEqual(path[-1], (0.5, 0.0))
        self.assertGreater(len(path), 2)
        self.assertTrue(any(abs(y) > 0.14 for _, y in path[1:-1]))


if __name__ == "__main__":
    unittest.main()
