import sys
import unittest
from pathlib import Path

CONTROLLER_DIR = Path(__file__).resolve().parents[1] / "controllers" / "epuck2_obstacle_avoidance"
if str(CONTROLLER_DIR) not in sys.path:
    sys.path.insert(0, str(CONTROLLER_DIR))

from obstacle_avoidance_logic import (  # noqa: E402
    AvoidanceState,
    MAX_SPEED,
    compute_wheel_speeds,
)


class EPuckObstacleAvoidanceLogicTests(unittest.TestCase):
    def test_clear_path_moves_forward(self):
        left, right = compute_wheel_speeds([0.0] * 8)
        self.assertGreater(left, 0.0)
        self.assertGreater(right, 0.0)
        self.assertAlmostEqual(left, right, places=6)

    def test_front_obstacle_turns_right_when_left_side_is_more_blocked(self):
        sensors = [1.0, 0.9, 0.8, 0.2, 0.0, 0.1, 0.2, 0.8]
        left, right = compute_wheel_speeds(sensors)
        self.assertGreater(left, right)
        self.assertGreaterEqual(left, 0.0)
        self.assertLessEqual(abs(left), MAX_SPEED)
        self.assertLessEqual(abs(right), MAX_SPEED)

    def test_front_obstacle_turns_left_when_right_side_is_more_blocked(self):
        sensors = [0.8, 0.2, 0.1, 0.0, 0.2, 0.8, 0.9, 1.0]
        left, right = compute_wheel_speeds(sensors)
        self.assertGreater(right, left)
        self.assertGreaterEqual(right, 0.0)
        self.assertLessEqual(abs(left), MAX_SPEED)
        self.assertLessEqual(abs(right), MAX_SPEED)

    def test_keeps_same_turn_direction_while_front_obstacle_persists(self):
        state = AvoidanceState()

        first_left, first_right = compute_wheel_speeds(
            [1.0, 0.9, 0.8, 0.2, 0.0, 0.1, 0.2, 0.8],
            state,
        )
        second_left, second_right = compute_wheel_speeds(
            [0.8, 0.2, 0.1, 0.0, 0.2, 0.8, 0.9, 1.0],
            state,
        )

        self.assertGreater(first_left, first_right)
        self.assertGreater(second_left, second_right)


if __name__ == "__main__":
    unittest.main()
