from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(package='aerofusion_ros2', executable='camera_monitor', output='screen'),
        Node(
            package='aerofusion_ros2',
            executable='camera_publisher',
            output='screen',
            parameters=[{'fps': 2.0, 'width': 320, 'height': 240}],
        ),
    ])
