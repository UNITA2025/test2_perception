from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='test_pkg',
            executable='test',
            name='test',
            parameters=[{

            }]
        ),

        Node(
            package='test_pkg',
            executable='test_visualizer',
            name='test_visualizer',
            parameters=[{

            }]
        ),

        Node(
            package='test_pkg',
            executable='test_planner',
            name='test_planner',
            parameters=[{

            }]
        ),

    ])