import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

cv2_stub = types.ModuleType('cv2')
cv2_stub.VideoCapture = lambda *args, **kwargs: None
cv2_stub.imencode = lambda *args, **kwargs: (False, None)
sys.modules.setdefault('cv2', cv2_stub)

geometry_msgs_module = types.ModuleType('geometry_msgs')
geometry_msgs_msg_module = types.ModuleType('geometry_msgs.msg')

class Twist:  # minimal test stub
    def __init__(self):
        self.linear = types.SimpleNamespace(x=0.0)
        self.angular = types.SimpleNamespace(z=0.0)

geometry_msgs_msg_module.Twist = Twist
geometry_msgs_module.msg = geometry_msgs_msg_module
sys.modules.setdefault('geometry_msgs', geometry_msgs_module)
sys.modules.setdefault('geometry_msgs.msg', geometry_msgs_msg_module)

rclpy_stub = types.ModuleType('rclpy')
rclpy_stub.init = lambda *args, **kwargs: None
rclpy_stub.ok = lambda: False
rclpy_stub.shutdown = lambda: None
rclpy_stub.spin = lambda *args, **kwargs: None
rclpy_stub.executors = types.SimpleNamespace(ExternalShutdownException=RuntimeError)
sys.modules.setdefault('rclpy', rclpy_stub)

rclpy_node_module = types.ModuleType('rclpy.node')
rclpy_node_module.Node = object
sys.modules.setdefault('rclpy.node', rclpy_node_module)

from epuck_swarm_control.epuck_team_dashboard import render_dashboard_html  # noqa: E402


class TeamWorkDashboardTests(unittest.TestCase):
    def test_dashboard_includes_start_system_and_webots_launch_controls(self):
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
                'webots_command': 'webots /tmp/demo.wbt',
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
                'webots_status': 'Webots not started.',
                'has_camera_frame': False,
            }
        )

        self.assertIn('Start System', html)
        self.assertIn("name='startup_command'", html)
        self.assertIn('Run Webots Simulation', html)
        self.assertIn("name='webots_command'", html)
        self.assertIn('Webots not started.', html)


if __name__ == '__main__':
    unittest.main()
