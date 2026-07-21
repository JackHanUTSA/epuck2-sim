import math
import sys
import tempfile
import unittest
from pathlib import Path

CONTROLLER_DIR = Path(__file__).resolve().parents[1] / "controllers" / "epuck2_obstacle_avoidance"
if str(CONTROLLER_DIR) not in sys.path:
    sys.path.insert(0, str(CONTROLLER_DIR))

from obstacle_mapping import (  # noqa: E402
    MappingState,
    Pose2D,
    mark_sensor_observations,
    save_map_image,
    update_pose,
)


class EPuckObstacleMappingTests(unittest.TestCase):
    def test_update_pose_moves_robot_forward(self):
        pose = Pose2D(x=0.0, y=0.0, theta=0.0)
        updated = update_pose(pose, left_speed=3.0, right_speed=3.0, dt=0.1)
        self.assertGreater(updated.x, pose.x)
        self.assertAlmostEqual(updated.y, 0.0, places=6)
        self.assertAlmostEqual(updated.theta, 0.0, places=6)

    def test_front_sensor_marks_occupied_cell_ahead(self):
        state = MappingState(width=80, height=80, meters_per_cell=0.01)
        pose = Pose2D(x=0.0, y=0.0, theta=0.0)
        mark_sensor_observations(state, pose, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        occupied_cells = [
            (x, y)
            for y, row in enumerate(state.grid)
            for x, value in enumerate(row)
            if value > state.UNKNOWN
        ]
        self.assertTrue(occupied_cells)
        self.assertTrue(any(x > state.origin_x for x, _ in occupied_cells))

    def test_save_map_image_writes_png(self):
        state = MappingState(width=40, height=40, meters_per_cell=0.01)
        pose = Pose2D(x=0.0, y=0.0, theta=0.0)
        mark_sensor_observations(state, pose, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "map.png"
            save_map_image(state, output_path)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
