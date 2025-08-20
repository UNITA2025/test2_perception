from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='sensor_initialize',
            executable='lidar_to_camera_broadcaster',
            name='lidar_camera_tf',
            parameters=[{
                'config_path': '/home/unita/test2_perception/src/sensor_initialize/config/'
            }]
        )
    ])
