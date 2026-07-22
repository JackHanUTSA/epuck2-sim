import math
from dataclasses import dataclass, field

MAX_SPEED = 6.28
BASE_SPEED = 0.5 * MAX_SPEED
FRONT_OBSTACLE_THRESHOLD = 0.12
TURN_SPEED = 0.45 * MAX_SPEED
TURN_LEFT = "left"
TURN_RIGHT = "right"
GOAL_TOLERANCE_M = 0.06
GOAL_HEADING_GAIN = 4.0
GOAL_MAX_TURN = 0.35 * MAX_SPEED
GOAL_BASE_SPEED = 0.55 * MAX_SPEED

# Braitenberg weight matrix: [sensor] -> (left_weight, right_weight)
WEIGHTS = [
    (-1.3, -1.0),
    (-1.3, -1.0),
    (-0.5, 0.5),
    (0.0, 0.0),
    (0.0, 0.0),
    (0.5, -0.5),
    (-1.0, -1.3),
    (-1.0, -1.3),
]

LEFT_BLOCKAGE_WEIGHTS = [1.0, 1.2, 1.0, 0.3]
RIGHT_BLOCKAGE_WEIGHTS = [0.3, 1.0, 1.2, 1.0]


@dataclass
class Pose2D:
    x: float
    y: float
    theta: float


@dataclass
class AvoidanceState:
    committed_turn: str | None = None


@dataclass
class WaypointNavigatorState:
    waypoints: list[tuple[float, float]] = field(default_factory=list)
    current_waypoint_index: int = 0
    completed_sweeps: int = 0
    max_steps_per_waypoint: int = 350
    steps_on_current_waypoint: int = 0

    def current_target(self) -> tuple[float, float]:
        if not self.waypoints:
            self.waypoints = build_lawnmower_waypoints()
        return self.waypoints[self.current_waypoint_index]

    def advance_waypoint(self):
        self.current_waypoint_index += 1
        self.steps_on_current_waypoint = 0
        if self.current_waypoint_index >= len(self.waypoints):
            self.current_waypoint_index = 0
            self.completed_sweeps += 1

    def advance_if_reached(self, pose, tolerance: float = GOAL_TOLERANCE_M) -> tuple[float, float]:
        target_x, target_y = self.current_target()
        distance = math.hypot(target_x - pose.x, target_y - pose.y)
        self.steps_on_current_waypoint += 1
        if distance <= tolerance or self.steps_on_current_waypoint > self.max_steps_per_waypoint:
            self.advance_waypoint()
        return self.current_target()


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def normalize_angle(theta):
    return math.atan2(math.sin(theta), math.cos(theta))


def build_lawnmower_waypoints(arena_half_extent: float = 0.36, lane_spacing: float = 0.18) -> list[tuple[float, float]]:
    positive_lanes = []
    lane = 0.0
    while lane <= arena_half_extent + 1e-9:
        positive_lanes.append(round(lane, 3))
        lane += lane_spacing

    lane_order: list[float] = []
    if positive_lanes:
        lane_order.append(0.0)
    for lane_y in reversed(positive_lanes[1:]):
        lane_order.append(lane_y)
        lane_order.append(-lane_y)

    waypoints: list[tuple[float, float]] = []
    left_x = -arena_half_extent
    right_x = arena_half_extent
    go_right = True
    for lane_y in lane_order:
        if go_right:
            waypoints.append((right_x, lane_y))
            waypoints.append((left_x, lane_y))
        else:
            waypoints.append((left_x, lane_y))
            waypoints.append((right_x, lane_y))
        go_right = not go_right
    return waypoints


def compute_side_blockage(normalized_sensors):
    left_blockage = sum(
        sensor_value * weight
        for sensor_value, weight in zip(normalized_sensors[:4], LEFT_BLOCKAGE_WEIGHTS)
    )
    right_blockage = sum(
        sensor_value * weight
        for sensor_value, weight in zip(normalized_sensors[4:], RIGHT_BLOCKAGE_WEIGHTS)
    )
    return left_blockage, right_blockage


def choose_turn_direction(normalized_sensors, state=None):
    if state and state.committed_turn is not None:
        return state.committed_turn

    left_blockage, right_blockage = compute_side_blockage(normalized_sensors)
    chosen_turn = TURN_RIGHT if left_blockage > right_blockage else TURN_LEFT

    if state:
        state.committed_turn = chosen_turn

    return chosen_turn


def choose_turn_speeds(normalized_sensors, state=None):
    chosen_turn = choose_turn_direction(normalized_sensors, state)

    if chosen_turn == TURN_LEFT:
        return -TURN_SPEED, TURN_SPEED

    return TURN_SPEED, -TURN_SPEED


def compute_goal_tracking_speeds(pose, target: tuple[float, float]):
    dx = target[0] - pose.x
    dy = target[1] - pose.y
    target_heading = math.atan2(dy, dx)
    heading_error = normalize_angle(target_heading - pose.theta)
    turn_adjust = clamp(heading_error * GOAL_HEADING_GAIN, -GOAL_MAX_TURN, GOAL_MAX_TURN)
    heading_scale = max(0.25, 1.0 - min(abs(heading_error), math.pi) / math.pi)
    forward_speed = GOAL_BASE_SPEED * heading_scale
    left_speed = clamp(forward_speed - turn_adjust, -MAX_SPEED, MAX_SPEED)
    right_speed = clamp(forward_speed + turn_adjust, -MAX_SPEED, MAX_SPEED)
    return left_speed, right_speed


def compute_braitenberg_speeds(normalized_sensors):
    left_speed = BASE_SPEED
    right_speed = BASE_SPEED

    for i, sensor_value in enumerate(normalized_sensors):
        left_speed += sensor_value * WEIGHTS[i][0] * MAX_SPEED
        right_speed += sensor_value * WEIGHTS[i][1] * MAX_SPEED

    return left_speed, right_speed


def compute_wheel_speeds(normalized_sensors, state=None, pose=None, navigation_state: WaypointNavigatorState | None = None):
    front_obstacle = max(
        normalized_sensors[0],
        normalized_sensors[1],
        normalized_sensors[6],
        normalized_sensors[7],
    )

    if front_obstacle >= FRONT_OBSTACLE_THRESHOLD:
        left_speed, right_speed = choose_turn_speeds(normalized_sensors, state)
    elif pose is not None and navigation_state is not None:
        if state:
            state.committed_turn = None
        target = navigation_state.advance_if_reached(pose)
        left_speed, right_speed = compute_goal_tracking_speeds(pose, target)
    else:
        if state:
            state.committed_turn = None
        left_speed, right_speed = compute_braitenberg_speeds(normalized_sensors)

    left_speed = clamp(left_speed, -MAX_SPEED, MAX_SPEED)
    right_speed = clamp(right_speed, -MAX_SPEED, MAX_SPEED)
    return left_speed, right_speed
