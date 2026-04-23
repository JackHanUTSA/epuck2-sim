"""
E-puck2 Obstacle Avoidance Controller (Python)
Swarmlab - Simple Braitenberg-style obstacle avoidance

Uses 8 IR proximity sensors (ps0-ps7) to drive the differential
wheels away from obstacles. Classic reactive behavior — no planning,
just sensor-motor coupling via a weight matrix.

Sensor layout (top view):
        ps0  ps7
      ps1      ps6
    ps2          ps5
      ps3      ps4

ps0/ps7 = front, ps2 = left, ps5 = right, ps3/ps4 = rear
"""

from controller import Robot

# --- Constants ---
TIME_STEP = 16          # ms, matches world basicTimeStep
MAX_SPEED = 6.28        # rad/s (e-puck max)
NUM_SENSORS = 8
SENSOR_NAMES = [f"ps{i}" for i in range(NUM_SENSORS)]

# Braitenberg weight matrix: [sensor] -> (left_weight, right_weight)
# Positive = excite that wheel, negative = inhibit
# Tuned so front/front-left obstacles steer right and vice versa
WEIGHTS = [
    (-1.3, -1.0),   # ps0 - front-right
    (-1.3, -1.0),   # ps1 - front-right-45
    (-0.5,  0.5),   # ps2 - right
    ( 0.0,  0.0),   # ps3 - rear-right
    ( 0.0,  0.0),   # ps4 - rear-left
    ( 0.5, -0.5),   # ps5 - left
    (-1.0, -1.3),   # ps6 - front-left-45
    (-1.0, -1.3),   # ps7 - front-left
]

# Base forward speed (fraction of MAX_SPEED)
BASE_SPEED = 0.5 * MAX_SPEED


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def main():
    robot = Robot()
    
    # --- Initialize proximity sensors ---
    sensors = []
    for name in SENSOR_NAMES:
        sensor = robot.getDevice(name)
        sensor.enable(TIME_STEP)
        sensors.append(sensor)
    
    # --- Initialize motors (velocity control) ---
    left_motor = robot.getDevice("left wheel motor")
    right_motor = robot.getDevice("right wheel motor")
    left_motor.setPosition(float('inf'))
    right_motor.setPosition(float('inf'))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)
    
    # --- Initialize LEDs ---
    leds = []
    for i in range(10):
        led = robot.getDevice(f"led{i}")
        if led:
            leds.append(led)
    
    print("[e-puck2] Obstacle avoidance controller started.")
    print(f"[e-puck2] Time step: {TIME_STEP}ms | Max speed: {MAX_SPEED} rad/s")
    
    step_count = 0
    
    # --- Main loop ---
    while robot.step(TIME_STEP) != -1:
        # Read sensor values (0-4096, higher = closer to obstacle)
        values = [s.getValue() for s in sensors]
        
        # Normalize to [0, 1]
        normalized = [v / 4096.0 for v in values]
        
        # Compute wheel speeds via Braitenberg
        left_speed = BASE_SPEED
        right_speed = BASE_SPEED
        
        for i in range(NUM_SENSORS):
            left_speed  += normalized[i] * WEIGHTS[i][0] * MAX_SPEED
            right_speed += normalized[i] * WEIGHTS[i][1] * MAX_SPEED
        
        # Clamp
        left_speed  = clamp(left_speed,  -MAX_SPEED, MAX_SPEED)
        right_speed = clamp(right_speed, -MAX_SPEED, MAX_SPEED)
        
        # Apply
        left_motor.setVelocity(left_speed)
        right_motor.setVelocity(right_speed)
        
        # Blink LEDs in sequence
        step_count += 1
        if leds:
            for j, led in enumerate(leds):
                led.set(1 if j == (step_count // 10) % len(leds) else 0)
        
        # Log every 2 seconds
        if step_count % (2000 // TIME_STEP) == 0:
            front_avg = (normalized[0] + normalized[7]) / 2.0
            print(f"[e-puck2] Step {step_count} | "
                  f"Front: {front_avg:.2f} | "
                  f"L: {left_speed:.2f} R: {right_speed:.2f}")


if __name__ == "__main__":
    main()
