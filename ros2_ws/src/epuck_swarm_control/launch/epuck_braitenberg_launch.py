from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    epuck_share = get_package_share_directory('webots_ros2_epuck')
    our_share = get_package_share_directory('epuck_swarm_control')

    robot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(epuck_share, 'launch', 'robot_launch.py')
        )
    )

    controller_node = Node(
        package='epuck_swarm_control',
        executable='epuck_braitenberg',
        output='screen',
        parameters=[{
            'forward_speed': 0.05,
            'turn_gain': 6.0,
            'front_threshold': 0.04,
            'avoid_gain': 1.0,
            'max_turn': 2.5,
            'timer_period': 0.05,
        }],
    )

    return LaunchDescription([
        robot_launch,
        controller_node,
    ])
