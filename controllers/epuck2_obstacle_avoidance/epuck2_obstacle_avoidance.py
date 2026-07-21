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

import os

from controller import Robot

from obstacle_avoidance_logic import AvoidanceState, MAX_SPEED, compute_wheel_speeds
from obstacle_mapping import MappingState, Pose2D, mark_sensor_observations, save_map_image, update_pose

# --- Constants ---
TIME_STEP = 16          # ms, matches world basicTimeStep
NUM_SENSORS = 8
SENSOR_NAMES = [f"ps{i}" for i in range(NUM_SENSORS)]
MAP_SAVE_EVERY_STEPS = 125
DEFAULT_MAP_OUTPUT = "/tmp/epuck2_obstacle_avoidance_map.png"


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
    avoidance_state = AvoidanceState()
    mapping_state = MappingState(width=160, height=160, meters_per_cell=0.01)
    pose = Pose2D(x=0.0, y=0.0, theta=0.0)
    map_output_path = os.environ.get("EPUCK_MAP_OUTPUT", DEFAULT_MAP_OUTPUT)
    save_period = int(os.environ.get("EPUCK_MAP_SAVE_EVERY_STEPS", str(MAP_SAVE_EVERY_STEPS)))
    dt = TIME_STEP / 1000.0
    print(f"[e-puck2] Mapping output: {map_output_path}")
    
    # --- Main loop ---
    while robot.step(TIME_STEP) != -1:
        # Read sensor values (0-4096, higher = closer to obstacle)
        values = [s.getValue() for s in sensors]
        
        # Normalize to [0, 1]
        normalized = [v / 4096.0 for v in values]

        # Compute wheel speeds. When a wall or box is detected ahead,
        # the helper compares left/right blockage and commits to the
        # chosen open side until the front clears to avoid flip-flopping.
        left_speed, right_speed = compute_wheel_speeds(normalized, avoidance_state)

        # Update the local occupancy map from the current pose and current
        # IR observations, then integrate wheel motion for the next step.
        mark_sensor_observations(mapping_state, pose, normalized)
        pose = update_pose(pose, left_speed, right_speed, dt)
        
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

        if save_period > 0 and step_count % save_period == 0:
            save_map_image(mapping_state, map_output_path, pose)
            print(f"[e-puck2] Saved occupancy map to {map_output_path}")

    save_map_image(mapping_state, map_output_path, pose)
    print(f"[e-puck2] Final occupancy map saved to {map_output_path}")


if __name__ == "__main__":
    main()
