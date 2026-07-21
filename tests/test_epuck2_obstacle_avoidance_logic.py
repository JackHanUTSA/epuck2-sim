import sys
import unittest
from pathlib import Path

CONTROLLER_DIR = Path(__file__).resolve().parents[1] / "controllers" / "epuck2_obstacle_avoidance"
if str(CONTROLLER_DIR) not in sys.path:
    sys.path.insert(0, str(CONTROLLER_DIR))

from obstacle_avoidance_logic import compute_wheel_speeds, MAX_SPEED  # noqa: E402


class EPuckObstacleAvoidanceLogicTests(unittest.TestCase):
    def test_clear_path_moves_forward(self):
        left, right = compute_wheel_speeds([0.0] * 8)
        self.assertGreater(left, 0.0)
        self.assertGreater(right, 0.0)
        self.assertAlmostEqual(left, right, places=6)

    def test_front_wall_forces_right_turn(self):
        sensors = [1.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.8, 1.0]
        left, right = compute_wheel_speeds(sensors)
        self.assertGreater(left, right)
        self.assertGreaterEqual(left, 0.0)
        self.assertLessEqual(abs(left), MAX_SPEED)
        self.assertLessEqual(abs(right), MAX_SPEED)


if __name__ == "__main__":
    unittest.main()
