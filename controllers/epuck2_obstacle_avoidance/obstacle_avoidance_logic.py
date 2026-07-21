from dataclasses import dataclass

MAX_SPEED = 6.28
BASE_SPEED = 0.5 * MAX_SPEED
FRONT_OBSTACLE_THRESHOLD = 0.12
TURN_SPEED = 0.45 * MAX_SPEED
TURN_LEFT = "left"
TURN_RIGHT = "right"

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
class AvoidanceState:
    committed_turn: str | None = None


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


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


def compute_wheel_speeds(normalized_sensors, state=None):
    left_speed = BASE_SPEED
    right_speed = BASE_SPEED

    for i, sensor_value in enumerate(normalized_sensors):
        left_speed += sensor_value * WEIGHTS[i][0] * MAX_SPEED
        right_speed += sensor_value * WEIGHTS[i][1] * MAX_SPEED

    front_obstacle = max(
        normalized_sensors[0],
        normalized_sensors[1],
        normalized_sensors[6],
        normalized_sensors[7],
    )

    if front_obstacle >= FRONT_OBSTACLE_THRESHOLD:
        left_speed, right_speed = choose_turn_speeds(normalized_sensors, state)
    elif state:
        state.committed_turn = None

    left_speed = clamp(left_speed, -MAX_SPEED, MAX_SPEED)
    right_speed = clamp(right_speed, -MAX_SPEED, MAX_SPEED)
    return left_speed, right_speed
