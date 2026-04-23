from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.dreamer_env import TeamState, build_observation, compute_reward
from src.dreamer_gym import DreamerTeamGymEnv, sample_waypoint


@dataclass(frozen=True)
class BridgePaths:
    root: Path
    action: Path
    reset: Path
    state: Path

    @classmethod
    def from_dir(cls, root: str | Path) -> "BridgePaths":
        root = Path(root)
        return cls(root=root, action=root / "action.json", reset=root / "reset.json", state=root / "state.json")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)


def snapshot_to_state(snapshot: dict[str, Any]) -> TeamState:
    robots = snapshot.get("robots") or []
    robot_poses = [tuple(robot["pose"]) for robot in robots]
    return TeamState(
        robot_poses=robot_poses,
        waypoint=tuple(snapshot.get("waypoint", (0.4, 0.0))),
        team_heading=float(snapshot.get("team_heading", 0.0)),
        prev_action=tuple(snapshot.get("prev_action", (0.0, 0.0))),
        step_count=int(snapshot.get("step_count", 0)),
    )


@dataclass
class FileBridgeClient:
    paths: BridgePaths
    world_path: Path | None = None
    mode: str = "auto"
    process: subprocess.Popen | None = field(default=None, init=False)
    _seq: int = field(default=0, init=False)

    def start(self) -> None:
        self.paths.root.mkdir(parents=True, exist_ok=True)
        if self.process is not None or self.world_path is None:
            return
        env = os.environ.copy()
        env["DREAMER_TEAM_MODE"] = "bridge"
        env["DREAMER_TEAM_BRIDGE_DIR"] = str(self.paths.root)
        cmd = build_webots_bridge_command(
            world_path=self.world_path,
            bridge_dir=self.paths.root,
            mode=self.mode,
        )
        self.process = subprocess.Popen(cmd, env=env, start_new_session=True)

    def send_reset(self, *, waypoint: tuple[float, float], seed: int | None = None) -> int:
        self._seq += 1
        _atomic_write_json(self.paths.reset, {"seq": self._seq, "waypoint": list(waypoint), "seed": seed})
        return self._seq

    def send_action(self, action: tuple[float, float]) -> int:
        self._seq += 1
        _atomic_write_json(self.paths.action, {"seq": self._seq, "action": [float(action[0]), float(action[1])]})
        return self._seq

    def wait_for_snapshot(self, timeout: float = 5.0, *, min_seq: int | None = None) -> dict[str, Any]:
        deadline = time.time() + timeout
        last_error: Exception | None = None
        while time.time() < deadline:
            try:
                if self.paths.state.exists():
                    payload = json.loads(self.paths.state.read_text())
                    payload_seq = int(payload.get("seq", -1))
                    if min_seq is None or payload_seq >= min_seq:
                        return payload
            except Exception as exc:  # noqa: BLE001
                last_error = exc
            time.sleep(0.05)
        if last_error:
            raise RuntimeError(f"snapshot read failed: {last_error}")
        raise TimeoutError(f"timed out waiting for snapshot at {self.paths.state}")

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=10)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                    self.process.wait(timeout=5)
                except ProcessLookupError:
                    pass
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait(timeout=5)
        self.process = None


@dataclass
class DreamerTeamLiveWebotsEnv(DreamerTeamGymEnv):
    bridge: Any = None
    snapshot_timeout: float = 5.0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if self.bridge is None:
            raise RuntimeError("bridge is required for live Webots env")
        self.bridge.start()
        waypoint = None
        if options and "waypoint" in options:
            waypoint = tuple(options["waypoint"])
        elif self.randomize_waypoint_on_reset:
            if seed is not None:
                waypoint = sample_waypoint(seed=seed, arena_half_extent=self.config.arena_half_extent, margin=0.12)
            else:
                waypoint = sample_waypoint(arena_half_extent=self.config.arena_half_extent, margin=0.12)
        else:
            waypoint = (0.4, 0.0)
        reset_seq = self.bridge.send_reset(waypoint=waypoint, seed=seed)
        snapshot = self.bridge.wait_for_snapshot(timeout=self.snapshot_timeout, min_seq=reset_seq)
        self.state = snapshot_to_state(snapshot)
        obs = build_observation(self.state)
        return obs, {"waypoint": self.state.waypoint, "state": self.state, "snapshot": snapshot}

    def step(self, action):
        if self.state is None:
            raise RuntimeError("reset() must be called before step()")
        action_seq = self.bridge.send_action((float(action[0]), float(action[1])))
        snapshot = self.bridge.wait_for_snapshot(timeout=self.snapshot_timeout, min_seq=action_seq)
        previous = self.state
        current = snapshot_to_state(snapshot)
        self.state = current
        reward = float(compute_reward(previous, current))
        centroid_x = sum(p[0] for p in current.robot_poses) / len(current.robot_poses)
        centroid_y = sum(p[1] for p in current.robot_poses) / len(current.robot_poses)
        dist = ((current.waypoint[0] - centroid_x) ** 2 + (current.waypoint[1] - centroid_y) ** 2) ** 0.5
        terminated = dist < self.config.reach_radius
        truncated = current.step_count >= self.max_steps
        obs = build_observation(current)
        info = {"state": current, "waypoint": current.waypoint, "distance_to_waypoint": dist, "snapshot": snapshot}
        return obs, reward, terminated, truncated, info

    def close(self):
        if self.bridge is not None:
            self.bridge.close()


def _reserve_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def build_webots_bridge_command(*, world_path: str | Path, bridge_dir: str | Path, mode: str = "auto") -> list[str]:
    del bridge_dir  # Bridge dir is passed via environment; kept here for API clarity.

    world_path = str(world_path)
    port = _reserve_free_tcp_port()
    base_args = [world_path, f"--port={port}", "--stdout", "--stderr", "--batch", "--mode=realtime", "--no-rendering", "--minimize"]
    if mode not in {"auto", "gui", "headless"}:
        raise ValueError(f"unsupported bridge launch mode: {mode}")

    if mode == "gui" or (mode == "auto" and os.environ.get("DISPLAY")):
        return ["webots", *base_args]

    webots_home = Path("/snap/webots/current/usr/share/webots")
    webots_bin = webots_home / "bin" / "webots-bin"
    headless_launcher = Path(__file__).resolve().parents[1] / "scripts" / "launch_webots_headless.sh"
    if webots_bin.exists() and headless_launcher.exists():
        return [str(headless_launcher), world_path, *base_args[1:]]

    return ["xvfb-run", "-a", "webots", *base_args]
