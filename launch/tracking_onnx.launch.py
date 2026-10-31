"""Optional: existing AeroDetect ONNX with ordered recorded images from a sequence."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, EmitEvent
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    publisher = Node(
        package='aerofusion_ros2', executable='recorded_image_publisher', output='screen',
        parameters=[{
            'image_dir': LaunchConfiguration('image_dir'),
            'fps': ParameterValue(LaunchConfiguration('fps'), value_type=float),
            'max_frames': ParameterValue(LaunchConfiguration('max_frames'), value_type=int),
        }],
    )
    tracker = Node(
        package='aerofusion_ros2', executable='tracking_node', output='screen',
        parameters=[{
            'mode': 'onnx',
            'reset_each_frame': ParameterValue(LaunchConfiguration('reset_each_frame'), value_type=bool),
            'aerodetect_root': LaunchConfiguration('aerodetect_root'),
            'onnx_path': LaunchConfiguration('onnx_path'),
        }],
    )
    return LaunchDescription([
        DeclareLaunchArgument('image_dir'),
        DeclareLaunchArgument('aerodetect_root'),
        DeclareLaunchArgument('onnx_path'),
        DeclareLaunchArgument('reset_each_frame', default_value='false'),
        DeclareLaunchArgument('fps', default_value='1.0'),
        DeclareLaunchArgument('max_frames', default_value='20'),
        tracker,
        publisher,
        RegisterEventHandler(OnProcessExit(target_action=publisher, on_exit=[EmitEvent(event=Shutdown(reason='Recorded image replay complete'))])),
    ])
