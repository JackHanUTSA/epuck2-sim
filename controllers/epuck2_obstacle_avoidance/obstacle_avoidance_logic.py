MAX_SPEED = 6.28
BASE_SPEED = 0.5 * MAX_SPEED
FRONT_WALL_THRESHOLD = 0.12
RIGHT_TURN_SPEED = 0.45 * MAX_SPEED

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


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def compute_wheel_speeds(normalized_sensors):
    left_speed = BASE_SPEED
    right_speed = BASE_SPEED

    for i, sensor_value in enumerate(normalized_sensors):
        left_speed += sensor_value * WEIGHTS[i][0] * MAX_SPEED
        right_speed += sensor_value * WEIGHTS[i][1] * MAX_SPEED

    front_wall = max(
        normalized_sensors[0],
        normalized_sensors[1],
        normalized_sensors[6],
        normalized_sensors[7],
    )

    if front_wall >= FRONT_WALL_THRESHOLD:
        left_speed = RIGHT_TURN_SPEED
        right_speed = -RIGHT_TURN_SPEED

    left_speed = clamp(left_speed, -MAX_SPEED, MAX_SPEED)
    right_speed = clamp(right_speed, -MAX_SPEED, MAX_SPEED)
    return left_speed, right_speed
