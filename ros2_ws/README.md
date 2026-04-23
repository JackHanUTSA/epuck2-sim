# ROS 2 e-puck2 Braitenberg Control

This workspace adds a ROS 2 node that reproduces simple reactive obstacle avoidance for the Webots e-puck2.

## Workspace

- `ros2_ws/src/epuck_swarm_control/`

## What it does

- launches `webots_ros2_epuck` demo world
- subscribes to `/ps0` ... `/ps7` (`sensor_msgs/Range`)
- publishes velocity commands on `/cmd_vel` (`geometry_msgs/Twist`)
- turns away from closer obstacles using a Braitenberg-style mapping

## Build

```bash
cd /home/jack/swarmlab/projects/epuck2-sim/ros2_ws
export PATH=/usr/bin:$PATH
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## Run

```bash
cd /home/jack/swarmlab/projects/epuck2-sim/ros2_ws
export PATH=/usr/bin:$PATH
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch epuck_swarm_control epuck_braitenberg_launch.py
```

## Team Work dashboard for 4 real e-puck2 robots

This workspace now also includes a ROS 2 dashboard that treats four e-puck2 robots as one team.

What it gives you:
- enter the name and IP address of each robot
- ping-check each robot so you can confirm they are reachable
- send one command to the whole team: forward, backward, left, right, arc left, arc right, stop
- start the whole system from the dashboard using a configurable local startup command
- optionally hook a top camera using an RTSP/HTTP stream URL or a local device index
- preview the most recent camera frame when the camera connection succeeds

Launch it:

```bash
cd /home/jack/swarmlab/projects/epuck2-sim/ros2_ws
export PATH=/usr/bin:$PATH
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch epuck_swarm_control epuck_team_work_launch.py
```

Open:

```bash
http://127.0.0.1:8080
```

Topic convention:
- the dashboard publishes team commands to `/<robot_name>/cmd_vel`
- use robot names that match your real ROS 2 namespaces, for example `epuck1` ... `epuck4`
- IP addresses are used for connection validation, while ROS topics are used for motion commands

Important:
- keep `PATH=/usr/bin:$PATH` before launching Webots ROS 2 packages on this machine
- this avoids picking up the Hermes Python 3.11 venv for `#!/usr/bin/env python3` ROS scripts
- ROS Humble here is built for Python 3.10

## Verified environment

- ROS 2 Humble installed
- `webots_ros2_epuck`, `webots_ros2_driver`, `webots_ros2_control`
- `ros2_control`, `ros2_controllers`, `controller_manager`
- topics confirmed: `/cmd_vel`, `/ps0`...`/ps7`, `/scan`, `/odom`
