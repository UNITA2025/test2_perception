# 음영구간 차선 인식하는 런치

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='lane_tracking_pkg',
            executable='yolov8_node',
            name='yolov8',
            parameters=[{

            }]
        ),

        Node(
            package='lane_tracking_pkg',
            executable='lane_info_extractor_node',
            name='lane_info_extractor',
            parameters=[{

            }]
        ),

        Node(
            package='lane_tracking_pkg',
            executable='yolov8_visualizer_node',
            name='yolov8_visualizer',
            parameters=[{

            }]
        ),

                Node(
            package='lane_tracking_pkg',
            executable='path_visualizer_node',
            name='path_visualizer',
            parameters=[{

            }]
        ),
    ])