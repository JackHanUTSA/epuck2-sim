import sys
import unittest
from pathlib import Path

CONTROLLER_DIR = Path(__file__).resolve().parents[1] / "controllers" / "epuck2_obstacle_avoidance"
if str(CONTROLLER_DIR) not in sys.path:
    sys.path.insert(0, str(CONTROLLER_DIR))

from obstacle_avoidance_logic import (  # noqa: E402
    AvoidanceState,
    MAX_SPEED,
    Pose2D,
    WaypointNavigatorState,
    build_lawnmower_waypoints,
    compute_wheel_speeds,
    compute_goal_tracking_speeds,
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

    def test_goal_tracking_turns_left_toward_upper_target(self):
        pose = Pose2D(x=0.0, y=0.0, theta=0.0)
        left, right = compute_goal_tracking_speeds(pose, (0.2, 0.2))
        self.assertGreater(right, left)
        self.assertGreater(right, 0.0)

    def test_navigation_waypoints_advance_after_target_reached(self):
        navigator = WaypointNavigatorState(waypoints=build_lawnmower_waypoints(arena_half_extent=0.3, lane_spacing=0.15))
        first_target = navigator.current_target()
        pose = Pose2D(x=first_target[0], y=first_target[1], theta=0.0)

        compute_wheel_speeds([0.0] * 8, pose=pose, navigation_state=navigator)

        self.assertNotEqual(navigator.current_target(), first_target)

    def test_navigation_waypoints_advance_after_timeout(self):
        navigator = WaypointNavigatorState(
            waypoints=[(0.3, 0.0), (-0.3, 0.0)],
            max_steps_per_waypoint=2,
        )
        pose = Pose2D(x=0.0, y=0.0, theta=0.0)
        first_target = navigator.current_target()

        compute_wheel_speeds([0.0] * 8, pose=pose, navigation_state=navigator)
        compute_wheel_speeds([0.0] * 8, pose=pose, navigation_state=navigator)
        compute_wheel_speeds([0.0] * 8, pose=pose, navigation_state=navigator)

        self.assertNotEqual(navigator.current_target(), first_target)


if __name__ == "__main__":
    unittest.main()
