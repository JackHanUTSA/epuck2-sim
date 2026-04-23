import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from epuck_swarm_control.epuck_team_dashboard import render_dashboard_html  # noqa: E402


class TeamWorkDashboardTests(unittest.TestCase):
    def test_dashboard_includes_start_system_button_and_command_field(self):
        html = render_dashboard_html(
            {
                'robots': [
                    {'name': 'epuck1', 'ip': '192.168.0.101'},
                    {'name': 'epuck2', 'ip': '192.168.0.102'},
                    {'name': 'epuck3', 'ip': '192.168.0.103'},
                    {'name': 'epuck4', 'ip': '192.168.0.104'},
                ],
                'camera_source': '',
                'startup_command': 'ros2 launch my_pkg bringup.launch.py',
                'last_command': 'stop',
                'errors': [],
                'connection_summary': {
                    'connected_count': 0,
                    'total_count': 4,
                    'team_ready': False,
                    'details': {},
                },
                'camera_status': 'Camera not checked.',
                'startup_status': 'System not started.',
                'has_camera_frame': False,
            }
        )

        self.assertIn('Start System', html)
        self.assertIn("name='startup_command'", html)
        self.assertIn('System not started.', html)


if __name__ == '__main__':
    unittest.main()
