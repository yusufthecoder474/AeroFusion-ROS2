"""Finite Stage-2 synthetic red-square tracking demo (no neural net)."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, EmitEvent
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    publisher = Node(
        package='aerofusion_ros2', executable='camera_publisher',
        output='screen', parameters=[{
            'fps': ParameterValue(LaunchConfiguration('fps'), value_type=float),
            'max_frames': ParameterValue(LaunchConfiguration('max_frames'), value_type=int),
            'width': 320, 'height': 240,
        }],
    )
    tracker = Node(package='aerofusion_ros2', executable='tracking_node',
                   output='screen', parameters=[{'mode': 'synthetic', 'min_iou': 0.3, 'max_misses': 2}])
    return LaunchDescription([
        DeclareLaunchArgument('fps', default_value='2.0'),
        DeclareLaunchArgument('max_frames', default_value='30'),
        tracker,
        publisher,
        RegisterEventHandler(OnProcessExit(target_action=publisher, on_exit=[EmitEvent(event=Shutdown(reason='30-frame demo complete'))])),
    ])
