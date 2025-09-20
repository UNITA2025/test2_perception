from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='info_publisher_pkg',
            executable='cone_info_node',
            name='cone_info_node',
            output='screen'
        ),
        Node(
            package='info_publisher_pkg',
            executable='drum_info_node',
            name='drum_info_node',
            output='screen'
        ),
    ])
