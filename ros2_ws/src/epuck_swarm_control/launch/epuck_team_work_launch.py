from launch import LaunchDescription
from launch_ros.actions import Node



def generate_launch_description():
    return LaunchDescription([
        Node(
            package='epuck_swarm_control',
            executable='epuck_team_dashboard',
            output='screen',
            parameters=[{
                'host': '0.0.0.0',
                'port': 8080,
                'linear_speed': 0.05,
                'angular_speed': 0.9,
            }],
        ),
    ])
