import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from epuck_swarm_control.team_work_logic import (  # noqa: E402
    TeamCommand,
    TeamInterfaceConfig,
    RobotConfig,
    build_team_command,
    summarize_connection_results,
    validate_robot_configs,
)


class TeamWorkLogicTests(unittest.TestCase):
    def test_validate_robot_configs_accepts_four_unique_ipv4_addresses(self):
        config = TeamInterfaceConfig(
            robots=[
                RobotConfig(name='epuck1', ip='192.168.0.101'),
                RobotConfig(name='epuck2', ip='192.168.0.102'),
                RobotConfig(name='epuck3', ip='192.168.0.103'),
                RobotConfig(name='epuck4', ip='192.168.0.104'),
            ],
            camera_source='rtsp://10.0.0.50/top_cam',
        )

        errors = validate_robot_configs(config)

        self.assertEqual(errors, [])

    def test_validate_robot_configs_rejects_duplicate_or_invalid_ips(self):
        config = TeamInterfaceConfig(
            robots=[
                RobotConfig(name='epuck1', ip='192.168.0.101'),
                RobotConfig(name='epuck2', ip='192.168.0.101'),
                RobotConfig(name='epuck3', ip='not_an_ip'),
                RobotConfig(name='epuck4', ip=''),
            ]
        )

        errors = validate_robot_configs(config)

        self.assertTrue(any('duplicate' in error.lower() for error in errors))
        self.assertTrue(any('valid ipv4' in error.lower() for error in errors))
        self.assertTrue(any('required' in error.lower() for error in errors))

    def test_build_team_command_maps_named_orders_to_velocity_commands(self):
        left = build_team_command('left', linear_speed=0.05, angular_speed=0.9)
        forward = build_team_command('forward', linear_speed=0.05, angular_speed=0.9)
        stop = build_team_command('stop', linear_speed=0.05, angular_speed=0.9)

        self.assertEqual(left, TeamCommand(linear_x=0.0, angular_z=0.9, label='left'))
        self.assertEqual(forward, TeamCommand(linear_x=0.05, angular_z=0.0, label='forward'))
        self.assertEqual(stop, TeamCommand(linear_x=0.0, angular_z=0.0, label='stop'))

    def test_summarize_connection_results_marks_team_ready_only_when_all_robots_reachable(self):
        ready = summarize_connection_results(
            {
                'epuck1': True,
                'epuck2': True,
                'epuck3': True,
                'epuck4': True,
            }
        )
        not_ready = summarize_connection_results(
            {
                'epuck1': True,
                'epuck2': False,
                'epuck3': True,
                'epuck4': True,
            }
        )

        self.assertTrue(ready.team_ready)
        self.assertEqual(ready.connected_count, 4)
        self.assertFalse(not_ready.team_ready)
        self.assertEqual(not_ready.connected_count, 3)


if __name__ == '__main__':
    unittest.main()
