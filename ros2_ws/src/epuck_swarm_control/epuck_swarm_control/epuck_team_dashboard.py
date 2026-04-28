from __future__ import annotations

import json
import os
import shlex
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import threading
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

from .team_work_logic import (
    RobotConfig,
    TeamInterfaceConfig,
    TeamCommand,
    build_team_command,
    summarize_connection_results,
    validate_robot_configs,
)


DEFAULT_ROBOTS = [
    RobotConfig(name='epuck1', ip='192.168.0.101'),
    RobotConfig(name='epuck2', ip='192.168.0.102'),
    RobotConfig(name='epuck3', ip='192.168.0.103'),
    RobotConfig(name='epuck4', ip='192.168.0.104'),
]

DEFAULT_WEBOTS_WORLD = Path(__file__).resolve().parents[4] / 'worlds/epuck2_dreamer_team_4_capture.wbt'


def default_webots_command() -> str:
    return f"webots {shlex.quote(str(DEFAULT_WEBOTS_WORLD))}"


class EpuckTeamDashboardNode(Node):
    def __init__(self) -> None:
        super().__init__('epuck_team_dashboard')
        self.host = str(self.declare_parameter('host', '0.0.0.0').value)
        self.port = int(self.declare_parameter('port', 8080).value)
        self.linear_speed = float(self.declare_parameter('linear_speed', 0.05).value)
        self.angular_speed = float(self.declare_parameter('angular_speed', 0.9).value)
        self.config_path = Path(os.path.expanduser(str(self.declare_parameter(
            'config_path', '~/.config/epuck_swarm_control/team_interface.json'
        ).value)))
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self.last_command = 'stop'
        self.last_errors: list[str] = []
        self.last_connection_summary = summarize_connection_results({robot.name: False for robot in DEFAULT_ROBOTS})
        self.last_camera_status = 'Camera not checked.'
        self._camera_frame_jpeg: bytes | None = None
        self.startup_process: subprocess.Popen[str] | None = None
        self.startup_status = 'System not started.'
        self.webots_process: subprocess.Popen[str] | None = None
        self.webots_status = 'Webots not started.'

        self.team_config = self._load_or_default_config()
        self.cmd_publishers: dict[str, Any] = {}
        self._refresh_publishers()

        self.server = ThreadingHTTPServer((self.host, self.port), self._build_handler())
        self.server.daemon_threads = True
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        self.get_logger().info(f'e-puck team dashboard running at http://{self.host}:{self.port}')

    def destroy_node(self) -> bool:
        self.server.shutdown()
        self.server.server_close()
        return super().destroy_node()

    def _load_or_default_config(self) -> TeamInterfaceConfig:
        if not self.config_path.exists():
            config = TeamInterfaceConfig(
                robots=list(DEFAULT_ROBOTS),
                camera_source='',
                startup_command='',
                webots_command=default_webots_command(),
            )
            self._save_config(config)
            return config
        try:
            payload = json.loads(self.config_path.read_text())
            robots = [RobotConfig(name=item['name'], ip=item['ip']) for item in payload.get('robots', [])]
            if not robots:
                robots = list(DEFAULT_ROBOTS)
            return TeamInterfaceConfig(
                robots=robots,
                camera_source=payload.get('camera_source', ''),
                startup_command=payload.get('startup_command', ''),
                webots_command=payload.get('webots_command', default_webots_command()),
            )
        except Exception as exc:
            self.get_logger().warning(f'Failed to load config {self.config_path}: {exc}')
            return TeamInterfaceConfig(
                robots=list(DEFAULT_ROBOTS),
                camera_source='',
                startup_command='',
                webots_command=default_webots_command(),
            )

    def _save_config(self, config: TeamInterfaceConfig) -> None:
        payload = {
            'robots': [asdict(robot) for robot in config.robots],
            'camera_source': config.camera_source,
            'startup_command': config.startup_command,
            'webots_command': config.webots_command,
        }
        self.config_path.write_text(json.dumps(payload, indent=2))

    def _refresh_publishers(self) -> None:
        with self._lock:
            publishers: dict[str, Any] = {}
            for robot in self.team_config.robots:
                topic = f'/{robot.name}/cmd_vel'
                publishers[robot.name] = self.create_publisher(Twist, topic, 10)
            self.cmd_publishers = publishers

    def apply_config_update(self, form_data: dict[str, str]) -> list[str]:
        robots: list[RobotConfig] = []
        for idx in range(1, 5):
            name = form_data.get(f'robot_{idx}_name', '').strip() or f'epuck{idx}'
            ip = form_data.get(f'robot_{idx}_ip', '').strip()
            robots.append(RobotConfig(name=name, ip=ip))
        config = TeamInterfaceConfig(
            robots=robots,
            camera_source=form_data.get('camera_source', '').strip(),
            startup_command=form_data.get('startup_command', '').strip(),
            webots_command=form_data.get('webots_command', '').strip() or default_webots_command(),
        )
        errors = validate_robot_configs(config)
        with self._lock:
            self.last_errors = errors
        if errors:
            return errors
        with self._lock:
            self.team_config = config
            self._save_config(config)
            self._refresh_publishers()
        return []

    def send_team_order(self, order: str) -> TeamCommand:
        command = build_team_command(order, linear_speed=self.linear_speed, angular_speed=self.angular_speed)
        msg = Twist()
        msg.linear.x = command.linear_x
        msg.angular.z = command.angular_z
        with self._lock:
            for publisher in self.cmd_publishers.values():
                publisher.publish(msg)
            self.last_command = command.label
        self.get_logger().info(f'Sent team order {command.label}: v={command.linear_x:.3f} w={command.angular_z:.3f}')
        return command

    def start_system(self) -> bool:
        with self._lock:
            command = self.team_config.startup_command.strip()
            running = self.startup_process is not None and self.startup_process.poll() is None
        if running:
            with self._lock:
                self.startup_status = 'System startup command is already running.'
            return True
        if not command:
            with self._lock:
                self.startup_status = 'No startup command configured.'
            return False
        try:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception as exc:
            with self._lock:
                self.startup_status = f'System start failed: {exc}'
            return False
        with self._lock:
            self.startup_process = process
            self.startup_status = f'System start launched: {command}'
        self.get_logger().info(f'Launched startup command: {command}')
        return True

    def start_webots(self) -> bool:
        with self._lock:
            command = self.team_config.webots_command.strip()
            running = self.webots_process is not None and self.webots_process.poll() is None
        if running:
            with self._lock:
                self.webots_status = 'Webots simulation is already running.'
            return True
        if not command:
            with self._lock:
                self.webots_status = 'No Webots command configured.'
            return False
        try:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception as exc:
            with self._lock:
                self.webots_status = f'Webots launch failed: {exc}'
            return False
        with self._lock:
            self.webots_process = process
            self.webots_status = f'Webots launch started: {command}'
        self.get_logger().info(f'Launched Webots command: {command}')
        return True

    def check_connections(self) -> dict[str, bool]:
        results: dict[str, bool] = {}
        with self._lock:
            robots = list(self.team_config.robots)
        for robot in robots:
            ok = subprocess.run(
                ['ping', '-c', '1', '-W', '1', robot.ip],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            ).returncode == 0
            results[robot.name] = ok
        summary = summarize_connection_results(results)
        with self._lock:
            self.last_connection_summary = summary
        return results

    def check_camera(self) -> bool:
        with self._lock:
            source = self.team_config.camera_source.strip()
        if not source:
            with self._lock:
                self.last_camera_status = 'Camera not configured.'
                self._camera_frame_jpeg = None
            return False
        capture_source: Any = int(source) if source.isdigit() else source
        cap = cv2.VideoCapture(capture_source)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            with self._lock:
                self.last_camera_status = f'Camera connection failed: {source}'
                self._camera_frame_jpeg = None
            return False
        success, encoded = cv2.imencode('.jpg', frame)
        with self._lock:
            if success:
                self._camera_frame_jpeg = encoded.tobytes()
            self.last_camera_status = f'Camera connected: {source}'
        return True

    def snapshot_state(self) -> dict[str, Any]:
        with self._lock:
            robots = [asdict(robot) for robot in self.team_config.robots]
            return {
                'robots': robots,
                'camera_source': self.team_config.camera_source,
                'startup_command': self.team_config.startup_command,
                'webots_command': self.team_config.webots_command,
                'last_command': self.last_command,
                'errors': list(self.last_errors),
                'connection_summary': {
                    'connected_count': self.last_connection_summary.connected_count,
                    'total_count': self.last_connection_summary.total_count,
                    'team_ready': self.last_connection_summary.team_ready,
                    'details': dict(self.last_connection_summary.details),
                },
                'camera_status': self.last_camera_status,
                'startup_status': self.startup_status,
                'webots_status': self.webots_status,
                'has_camera_frame': self._camera_frame_jpeg is not None,
            }

    def get_camera_frame(self) -> bytes | None:
        with self._lock:
            return self._camera_frame_jpeg

    def _build_handler(self):
        node = self

        class DashboardHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                if parsed.path == '/camera.jpg':
                    frame = node.get_camera_frame()
                    if frame is None:
                        self.send_error(HTTPStatus.NOT_FOUND, 'No camera frame available')
                        return
                    self.send_response(HTTPStatus.OK)
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(frame)))
                    self.end_headers()
                    self.wfile.write(frame)
                    return
                html = render_dashboard_html(node.snapshot_state())
                body = html.encode('utf-8')
                self.send_response(HTTPStatus.OK)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:
                length = int(self.headers.get('Content-Length', '0'))
                raw = self.rfile.read(length).decode('utf-8')
                form = {key: values[0] for key, values in parse_qs(raw, keep_blank_values=True).items()}
                parsed = urlparse(self.path)
                if parsed.path == '/save':
                    node.apply_config_update(form)
                elif parsed.path == '/check':
                    node.apply_config_update(form)
                    node.check_connections()
                    node.check_camera()
                elif parsed.path == '/start':
                    node.apply_config_update(form)
                    node.start_system()
                elif parsed.path == '/start_webots':
                    node.apply_config_update(form)
                    node.start_webots()
                elif parsed.path == '/command':
                    node.apply_config_update(form)
                    order = form.get('order', 'stop')
                    node.send_team_order(order)
                else:
                    self.send_error(HTTPStatus.NOT_FOUND, 'Unknown action')
                    return
                self.send_response(HTTPStatus.SEE_OTHER)
                self.send_header('Location', '/')
                self.end_headers()

            def log_message(self, format: str, *args: Any) -> None:
                node.get_logger().debug(format % args)

        return DashboardHandler


def render_dashboard_html(state: dict[str, Any]) -> str:
    summary = state['connection_summary']
    robot_rows = []
    for idx, robot in enumerate(state['robots'], start=1):
        connected = summary['details'].get(robot['name'], False)
        status = 'Connected' if connected else 'Not checked / offline'
        robot_rows.append(
            f"<tr><td>{idx}</td><td><input name='robot_{idx}_name' value='{robot['name']}'/></td>"
            f"<td><input name='robot_{idx}_ip' value='{robot['ip']}'/></td><td>{status}</td></tr>"
        )
    error_html = ''.join(f'<li>{error}</li>' for error in state['errors']) or '<li>No validation errors.</li>'
    camera_block = (
        "<div><img src='/camera.jpg' alt='camera frame' style='max-width:480px;border:1px solid #999;'/></div>"
        if state['has_camera_frame'] else ''
    )
    return f"""
<!doctype html>
<html>
<head>
  <meta charset='utf-8'/>
  <title>e-puck2 Team Work Dashboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #f6f8fb; color: #14213d; }}
    table {{ border-collapse: collapse; width: 100%; background: white; }}
    th, td {{ border: 1px solid #ccd3e0; padding: 8px; text-align: left; }}
    .panel {{ background: white; border: 1px solid #ccd3e0; padding: 16px; margin-bottom: 18px; border-radius: 10px; }}
    .buttons button {{ margin-right: 10px; margin-bottom: 10px; padding: 10px 16px; }}
    input {{ width: 100%; padding: 6px; box-sizing: border-box; }}
  </style>
</head>
<body>
  <h1>e-puck2 Team Work Dashboard</h1>
  <div class='panel'>
    <p><strong>Group ready:</strong> {'YES' if summary['team_ready'] else 'NO'} ({summary['connected_count']}/{summary['total_count']} reachable)</p>
    <p><strong>Last team order:</strong> {state['last_command']}</p>
    <p><strong>Camera status:</strong> {state['camera_status']}</p>
    <p><strong>Startup status:</strong> {state['startup_status']}</p>
    <p><strong>Webots status:</strong> {state['webots_status']}</p>
  </div>

  <form method='post'>
    <div class='panel'>
      <h2>Robot IP setup</h2>
      <table>
        <tr><th>#</th><th>Robot name</th><th>IP address</th><th>Status</th></tr>
        {''.join(robot_rows)}
      </table>
      <p><strong>Top camera source</strong> (RTSP/HTTP URL or device index)</p>
      <input name='camera_source' value='{state['camera_source']}' placeholder='rtsp://... or 0'/>
      <p><strong>System startup command</strong> (local shell command for full bring-up)</p>
      <input name='startup_command' value='{state['startup_command']}' placeholder='ros2 launch your_pkg your_bringup.launch.py'/>
      <p><strong>Webots simulation command</strong> (local shell command to launch the simulation)</p>
      <input name='webots_command' value='{state['webots_command']}' placeholder='webots /path/to/world.wbt'/>
      <p>Use names that match your ROS namespaces, for example epuck1, epuck2, epuck3, epuck4.</p>
      <button formaction='/save' type='submit'>Save configuration</button>
      <button formaction='/check' type='submit'>Check robots + camera</button>
      <button formaction='/start' type='submit'>Start System</button>
      <button formaction='/start_webots' type='submit'>Run Webots Simulation</button>
    </div>

    <div class='panel'>
      <h2>Team motion commands</h2>
      <div class='buttons'>
        <button formaction='/command' name='order' value='forward' type='submit'>Forward</button>
        <button formaction='/command' name='order' value='backward' type='submit'>Backward</button>
        <button formaction='/command' name='order' value='left' type='submit'>Turn left</button>
        <button formaction='/command' name='order' value='right' type='submit'>Turn right</button>
        <button formaction='/command' name='order' value='spin_left' type='submit'>Arc left</button>
        <button formaction='/command' name='order' value='spin_right' type='submit'>Arc right</button>
        <button formaction='/command' name='order' value='stop' type='submit'>Stop</button>
      </div>
      <p>These commands publish the same team order to /&lt;robot_name&gt;/cmd_vel for each configured e-puck2.</p>
    </div>
  </form>

  <div class='panel'>
    <h2>Validation</h2>
    <ul>{error_html}</ul>
    {camera_block}
  </div>
</body>
</html>
"""


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = EpuckTeamDashboardNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            stop = Twist()
            for publisher in node.cmd_publishers.values():
                try:
                    publisher.publish(stop)
                except Exception:
                    pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
