from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="tl_perception_pkg",
            executable="tl_detector_node",
            name="tl_detector",
            parameters=[{
                "image_topic": "camera_front_up/image_raw",       # 퍼블리셔 토픽과 동일
                "model_path":  "traffic_light.pt",
                "conf_th":     0.3
            }]
        )
    ])