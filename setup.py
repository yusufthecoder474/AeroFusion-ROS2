from setuptools import setup
from glob import glob
import os

package_name = 'aerofusion_ros2'
setup(
    name=package_name,
    version='0.3.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Mohammad Yusuf K A',
    maintainer_email='mohammadyusuf.ka509@gmail.com',
    description='ROS 2 synthetic or ONNX object detection with simple IoU tracking',
    license='MIT',
    entry_points={'console_scripts': [
        'camera_publisher = aerofusion_ros2.image_publisher:main',
        'camera_monitor = aerofusion_ros2.image_monitor:main',
        'tracking_node = aerofusion_ros2.tracking_node:main',
        'recorded_image_publisher = aerofusion_ros2.recorded_image_publisher:main',
    ]},
)
