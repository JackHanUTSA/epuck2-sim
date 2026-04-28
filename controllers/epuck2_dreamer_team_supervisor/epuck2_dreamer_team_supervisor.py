from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

from controller import Supervisor

CURRENT_DIR = Path(__file__).resolve().parent
CONTROLLERS_DIR = CURRENT_DIR.parent
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

TIME_STEP = 16
MAX_SPEED = 6.28
ROBOT_NAMES = ["epuck2_A", "epuck2_B", "epuck2_C", "epuck2_D"]
ROLE_BY_NAME = dict(zip(ROBOT_NAMES, ROLE_ORDER))
SLOT_SPACING = 0.06
WAYPOINTS = [
    (0.0, 0.0),
    (0.34, 0.0),
    (0.34, 0.34),
    (0.0, 0.34),
    (-0.34, 0.34),
    (-0.34, 0.0),
    (-0.34, -0.34),
    (0.0, -0.34),
    (0.34, -0.34),
    (0.0, 0.0),
]
WAYPOINT_REACHED_RADIUS = 0.07
PATH_SUBGOAL_RADIUS = 0.05
LOG_PERIOD_STEPS = 80
ARENA_HALF_EXTENT = 0.70
PATH_GRID_RESOLUTION = 0.04
TEAM_FORMATION_RADIUS = 0.12
AUTONOMOUS_LOOKAHEAD_STEP = 0.035
OBSTACLE_BOXES = [
    ObstacleBox(center=(0.0, 0.22), size=(0.08, 0.08), yaw=0.3),
    ObstacleBox(center=(-0.22, -0.10), size=(0.08, 0.08), yaw=1.05),
    ObstacleBox(center=(0.24, -0.24), size=(0.08, 0.08), yaw=0.7),
]
BRIDGE_MAX_FORWARD_SPEED = 0.22
BRIDGE_MAX_TURN_RATE = 1.4
BRIDGE_DEFAULT_WAYPOINT = (0.4, 0.0)
BRIDGE_SPAWN_CENTROID = (0.0, 0.0)
BRIDGE_ROLE_OFFSETS = {
    "front_left": (-0.06, 0.06),
    "front_right": (0.06, 0.06),
    "rear_left": (-0.06, -0.06),
    "rear_right": (0.06, -0.06),
}


def node_pose(node) -> tuple[float, float, float]:
    translation = node.getPosition()
    orientation = node.getOrientation()
    heading = math.atan2(orientation[3], orientation[0])
    return translation[0], translation[1], heading


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)


def _set_robot_pose(node, *, x: float, y: float, heading: float) -> None:
    node.getField("translation").setSFVec3f([x, y, 0.0])
    node.getField("rotation").setSFRotation([0.0, 0.0, 1.0, heading])


def _apply_role_spawn(nodes: dict[str, object], *, centroid: tuple[float, float], heading: float) -> None:
    c = math.cos(heading)
    s = math.sin(heading)
    for name, role in ROLE_BY_NAME.items():
        ox, oy = BRIDGE_ROLE_OFFSETS[role]
        x = centroid[0] + ox * c - oy * s
        y = centroid[1] + ox * s + oy * c
        _set_robot_pose(nodes[name], x=x, y=y, heading=heading)


def _write_member_commands(
    *,
    custom_fields: dict[str, object],
    poses: dict[str, tuple[float, float, float]],
    centroid: tuple[float, float],
    team_heading: float,
    waypoint: tuple[float, float],
    phase_offset: int = 0,
) -> None:
    for phase, (name, pose) in enumerate(poses.items()):
        role = ROLE_BY_NAME[name]
        target = role_target_position(
            role,
            centroid=centroid,
            team_heading=team_heading,
            spacing=SLOT_SPACING,
        )
        left, right = tracking_command(
            pose=pose,
            target=target,
            max_speed=MAX_SPEED,
        )
        custom_fields[name].setSFString(
            json.dumps(
                {
                    "role": role,
                    "left": round(left, 5),
                    "right": round(right, 5),
                    "target_x": round(target[0], 5),
                    "target_y": round(target[1], 5),
                    "centroid_x": round(centroid[0], 5),
                    "centroid_y": round(centroid[1], 5),
                    "waypoint_x": round(waypoint[0], 5),
                    "waypoint_y": round(waypoint[1], 5),
                    "heading": round(team_heading, 5),
                    "phase": phase + phase_offset,
                }
            )
        )


def _write_bridge_snapshot(
    *,
    state_path: Path,
    seq: int,
    poses: dict[str, tuple[float, float, float]],
    waypoint: tuple[float, float],
    team_heading: float,
    prev_action: tuple[float, float],
    step_count: int,
) -> None:
    payload = {
        "seq": int(seq),
        "step_count": step_count,
        "waypoint": [float(waypoint[0]), float(waypoint[1])],
        "team_heading": float(team_heading),
        "prev_action": [float(prev_action[0]), float(prev_action[1])],
        "robots": [
            {"name": name, "pose": [float(pose[0]), float(pose[1]), float(pose[2])]} for name, pose in poses.items()
        ],
    }
    _atomic_write_json(state_path, payload)


def _build_team_route(start: tuple[float, float], goal: tuple[float, float]) -> list[tuple[float, float]]:
    return plan_team_path(
        start=start,
        goal=goal,
        obstacles=OBSTACLE_BOXES,
        arena_half_extent=ARENA_HALF_EXTENT,
        formation_radius=TEAM_FORMATION_RADIUS,
        grid_resolution=PATH_GRID_RESOLUTION,
    )


def run_autonomous(supervisor: Supervisor, nodes: dict[str, object], custom_fields: dict[str, object]) -> None:
    waypoint_index = 0
    step_count = 0
    poses = {name: node_pose(node) for name, node in nodes.items()}
    centroid = compute_centroid(poses.values())
    route = _build_team_route(centroid, WAYPOINTS[waypoint_index])
    route_index = 1 if len(route) > 1 else 0
    print("[dreamer_team] autonomous supervisor started for four-e-puck virtual body")
    print(f"[dreamer_team] initial planned route to waypoint {waypoint_index}: {route}")
    while supervisor.step(TIME_STEP) != -1:
        step_count += 1
        poses = {name: node_pose(node) for name, node in nodes.items()}
        centroid = compute_centroid(poses.values())
        waypoint = WAYPOINTS[waypoint_index]
        distance_to_waypoint = math.hypot(waypoint[0] - centroid[0], waypoint[1] - centroid[1])
        if distance_to_waypoint < WAYPOINT_REACHED_RADIUS:
            waypoint_index = (waypoint_index + 1) % len(WAYPOINTS)
            waypoint = WAYPOINTS[waypoint_index]
            route = _build_team_route(centroid, waypoint)
            route_index = 1 if len(route) > 1 else 0
            print(f"[dreamer_team] replanned route for waypoint {waypoint_index}: {route}")

        if route_index < len(route) - 1:
            subgoal = route[route_index]
            if math.hypot(subgoal[0] - centroid[0], subgoal[1] - centroid[1]) < PATH_SUBGOAL_RADIUS:
                route_index += 1
        subgoal = route[min(route_index, len(route) - 1)]

        dx = subgoal[0] - centroid[0]
        dy = subgoal[1] - centroid[1]
        distance_to_subgoal = math.hypot(dx, dy)
        team_heading = math.atan2(dy, dx) if (abs(dx) + abs(dy)) > 1e-9 else 0.0
        step_distance = min(AUTONOMOUS_LOOKAHEAD_STEP, distance_to_subgoal)
        target_centroid = (
            centroid[0] + math.cos(team_heading) * step_distance,
            centroid[1] + math.sin(team_heading) * step_distance,
        )
        _write_member_commands(
            custom_fields=custom_fields,
            poses=poses,
            centroid=target_centroid,
            team_heading=team_heading,
            waypoint=waypoint,
        )

        if step_count % LOG_PERIOD_STEPS == 0:
            heading_errors = [normalize_angle(team_heading - pose[2]) for pose in poses.values()]
            print(
                "[dreamer_team] "
                f"step={step_count} centroid=({centroid[0]:.2f},{centroid[1]:.2f}) "
                f"goal=({waypoint[0]:.2f},{waypoint[1]:.2f}) subgoal=({subgoal[0]:.2f},{subgoal[1]:.2f}) "
                f"route_idx={route_index}/{max(0, len(route)-1)} d_goal={distance_to_waypoint:.2f} "
                f"heading_err_avg={sum(abs(v) for v in heading_errors)/len(heading_errors):.2f}"
            )


def run_bridge(supervisor: Supervisor, nodes: dict[str, object], custom_fields: dict[str, object]) -> None:
    bridge_dir = Path(os.environ.get("DREAMER_TEAM_BRIDGE_DIR", "/tmp/dreamer_team_bridge"))
    action_path = bridge_dir / "action.json"
    reset_path = bridge_dir / "reset.json"
    state_path = bridge_dir / "state.json"
    bridge_dir.mkdir(parents=True, exist_ok=True)

    action_seq = -1
    reset_seq = -1
    snapshot_seq = 0
    prev_action = (0.0, 0.0)
    current_waypoint = BRIDGE_DEFAULT_WAYPOINT
    virtual_heading = 0.0
    episode_step_count = 0

    _apply_role_spawn(nodes, centroid=BRIDGE_SPAWN_CENTROID, heading=0.0)
    supervisor.simulationResetPhysics()
    print(f"[dreamer_team] bridge supervisor started | dir={bridge_dir}")

    while supervisor.step(TIME_STEP) != -1:
        reset_received = False
        action_received = False

        reset_payload = _read_json(reset_path)
        if reset_payload is not None and int(reset_payload.get("seq", -1)) > reset_seq:
            reset_seq = int(reset_payload["seq"])
            snapshot_seq = reset_seq
            waypoint = reset_payload.get("waypoint", list(BRIDGE_DEFAULT_WAYPOINT))
            current_waypoint = (float(waypoint[0]), float(waypoint[1]))
            prev_action = (0.0, 0.0)
            virtual_heading = 0.0
            episode_step_count = 0
            _apply_role_spawn(nodes, centroid=BRIDGE_SPAWN_CENTROID, heading=0.0)
            supervisor.simulationResetPhysics()
            reset_received = True

        action_payload = _read_json(action_path)
        if action_payload is not None and int(action_payload.get("seq", -1)) > action_seq:
            action_seq = int(action_payload["seq"])
            snapshot_seq = action_seq
            action = action_payload.get("action", [0.0, 0.0])
            prev_action = (float(action[0]), float(action[1]))
            action_received = True

        poses = {name: node_pose(node) for name, node in nodes.items()}
        centroid = compute_centroid(poses.values())

        if action_received:
            dt = TIME_STEP / 1000.0
            virtual_heading = normalize_angle(virtual_heading + prev_action[1] * BRIDGE_MAX_TURN_RATE * dt)
            target_centroid = (
                centroid[0] + math.cos(virtual_heading) * prev_action[0] * BRIDGE_MAX_FORWARD_SPEED * dt,
                centroid[1] + math.sin(virtual_heading) * prev_action[0] * BRIDGE_MAX_FORWARD_SPEED * dt,
            )
            episode_step_count += 1
        else:
            target_centroid = centroid

        _write_member_commands(
            custom_fields=custom_fields,
            poses=poses,
            centroid=target_centroid,
            team_heading=virtual_heading,
            waypoint=current_waypoint,
            phase_offset=episode_step_count,
        )

        if reset_received or action_received:
            _write_bridge_snapshot(
                state_path=state_path,
                seq=snapshot_seq,
                poses=poses,
                waypoint=current_waypoint,
                team_heading=virtual_heading,
                prev_action=prev_action,
                step_count=episode_step_count,
            )


def main() -> None:
    supervisor = Supervisor()
    nodes = {name: supervisor.getFromDef(name) for name in ROBOT_NAMES}
    if any(node is None for node in nodes.values()):
        missing = [name for name, node in nodes.items() if node is None]
        raise RuntimeError(f"missing robot defs: {missing}")

    custom_fields = {name: node.getField("customData") for name, node in nodes.items()}
    mode = os.environ.get("DREAMER_TEAM_MODE", "autonomous").strip().lower()
    if mode == "bridge":
        run_bridge(supervisor, nodes, custom_fields)
        return
    run_autonomous(supervisor, nodes, custom_fields)


if __name__ == "__main__":
    main()
