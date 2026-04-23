from __future__ import annotations

import json
import sys
from pathlib import Path

from controller import Robot

CURRENT_DIR = Path(__file__).resolve().parent
CONTROLLERS_DIR = CURRENT_DIR.parent
if str(CONTROLLERS_DIR) not in sys.path:
    sys.path.insert(0, str(CONTROLLERS_DIR))

from dreamer_team_common import blend_obstacle_avoidance, clamp  # noqa: E402

TIME_STEP = 16
MAX_SPEED = 6.28
SENSOR_NAMES = [f"ps{i}" for i in range(8)]
LOG_PERIOD_STEPS = 200


def parse_command(custom_data: str) -> dict:
    if not custom_data:
        return {}
    try:
        return json.loads(custom_data)
    except json.JSONDecodeError:
        return {}


def main() -> None:
    robot = Robot()
    name = robot.getName()

    sensors = []
    for sensor_name in SENSOR_NAMES:
        sensor = robot.getDevice(sensor_name)
        sensor.enable(TIME_STEP)
        sensors.append(sensor)

    left_motor = robot.getDevice("left wheel motor")
    right_motor = robot.getDevice("right wheel motor")
    left_motor.setPosition(float("inf"))
    right_motor.setPosition(float("inf"))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    leds = []
    for i in range(10):
        try:
            leds.append(robot.getDevice(f"led{i}"))
        except BaseException:
            break

    print(f"[{name}] dreamer-team member controller started")
    step_count = 0
    while robot.step(TIME_STEP) != -1:
        step_count += 1
        command = parse_command(robot.getCustomData())
        desired_left = float(command.get("left", 0.0))
        desired_right = float(command.get("right", 0.0))
        role = command.get("role", "unknown")

        sensor_values = [sensor.getValue() for sensor in sensors]
        left_speed, right_speed = blend_obstacle_avoidance(
            desired_left=desired_left,
            desired_right=desired_right,
            sensor_values=sensor_values,
            max_speed=MAX_SPEED,
        )
        left_speed = clamp(left_speed, -MAX_SPEED, MAX_SPEED)
        right_speed = clamp(right_speed, -MAX_SPEED, MAX_SPEED)
        left_motor.setVelocity(left_speed)
        right_motor.setVelocity(right_speed)

        if leds:
            index = int(command.get("phase", 0)) % len(leds)
            for i, led in enumerate(leds):
                led.set(1 if i == index else 0)

        if step_count % LOG_PERIOD_STEPS == 0:
            print(
                f"[{name}] role={role} step={step_count} desired=({desired_left:.2f},{desired_right:.2f}) "
                f"applied=({left_speed:.2f},{right_speed:.2f})"
            )


if __name__ == "__main__":
    main()
