"""
E-puck2 basic swarm controller for Webots.

Behavior:
- drive forward by default
- avoid nearby robots / walls / obstacles with proximity sensors
- add a small robot-specific turn bias so the swarm does not stay symmetric
- blink LEDs with a robot-specific phase for easy visual distinction

Designed to be attached to multiple e-puck2 robots in the same world.
"""

from controller import Robot

TIME_STEP = 16
MAX_SPEED = 6.28
NUM_SENSORS = 8
SENSOR_NAMES = [f"ps{i}" for i in range(NUM_SENSORS)]

# Reactive weights. Front sensors inhibit forward motion strongly;
# side sensors bias turning away from nearby objects.
WEIGHTS = [
    (-1.20, -0.90),  # ps0 front-right
    (-1.05, -0.75),  # ps1 right-front
    (-0.35,  0.55),  # ps2 right
    (-0.10,  0.15),  # ps3 rear-right
    ( 0.15, -0.10),  # ps4 rear-left
    ( 0.55, -0.35),  # ps5 left
    (-0.75, -1.05),  # ps6 left-front
    (-0.90, -1.20),  # ps7 front-left
]

BASE_SPEED = 0.65 * MAX_SPEED
TURN_BIAS_GAIN = 0.18 * MAX_SPEED
LOG_PERIOD_STEPS = 250  # ~4 s


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def deterministic_bias(name: str) -> float:
    # Stable per-robot bias in [-1, 1]. Helps robots break symmetry.
    total = sum(ord(ch) for ch in name)
    bucket = total % 11  # 0..10
    return (bucket - 5) / 5.0


def main():
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
        led = robot.getDevice(f"led{i}")
        if led:
            leds.append(led)

    bias = deterministic_bias(name)
    print(f"[{name}] basic swarm controller started | bias={bias:+.2f}")

    step_count = 0
    while robot.step(TIME_STEP) != -1:
        step_count += 1
        values = [sensor.getValue() for sensor in sensors]
        normalized = [min(1.0, value / 4096.0) for value in values]

        left_speed = BASE_SPEED
        right_speed = BASE_SPEED

        for i in range(NUM_SENSORS):
            left_speed += normalized[i] * WEIGHTS[i][0] * MAX_SPEED
            right_speed += normalized[i] * WEIGHTS[i][1] * MAX_SPEED

        # Mild persistent asymmetry so the robots don't mirror each other forever.
        left_speed += bias * TURN_BIAS_GAIN
        right_speed -= bias * TURN_BIAS_GAIN

        # If front is very crowded, force a stronger turn.
        front_right = max(normalized[0], normalized[1])
        front_left = max(normalized[6], normalized[7])
        front_max = max(front_right, front_left)
        if front_max > 0.55:
            if front_left > front_right:
                left_speed += 0.20 * MAX_SPEED
                right_speed -= 0.35 * MAX_SPEED
            else:
                left_speed -= 0.35 * MAX_SPEED
                right_speed += 0.20 * MAX_SPEED

        left_speed = clamp(left_speed, -MAX_SPEED, MAX_SPEED)
        right_speed = clamp(right_speed, -MAX_SPEED, MAX_SPEED)

        left_motor.setVelocity(left_speed)
        right_motor.setVelocity(right_speed)

        if leds:
            phase = (step_count // 8 + int((bias + 1.0) * 4)) % len(leds)
            for index, led in enumerate(leds):
                led.set(1 if index == phase else 0)

        if step_count % LOG_PERIOD_STEPS == 0:
            left_avg = (normalized[5] + normalized[6] + normalized[7]) / 3.0
            right_avg = (normalized[0] + normalized[1] + normalized[2]) / 3.0
            print(
                f"[{name}] step={step_count} front={front_max:.2f} "
                f"left={left_avg:.2f} right={right_avg:.2f} "
                f"motors=({left_speed:.2f}, {right_speed:.2f})"
            )


if __name__ == "__main__":
    main()
