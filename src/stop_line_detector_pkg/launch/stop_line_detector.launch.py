
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 비디오 퍼블리셔
        # Node(
        #     package="stop_line_detector_pkg",
        #     executable="video_publisher_node",
        #     name="video_pub",
        #     parameters=[{
        #         "video_path": "/home/unita/test2_perception/src/stop_line_detector_pkg/video/18.mp4",
        #         "fps": 30.0,
        #     }],
        #     output="screen",
        # ),
        # 정지선 검출기
        Node(
            package="stop_line_detector_pkg",
            executable="stop_line_detector_node",
            name="stop_line_detector",
            parameters=[{
                "model_path": "stop_line.pt",
                "image_topic": "/camera_front_down/image_raw",    # 퍼블리셔와 일치
                "conf_thres": 0.25,
                "iou_thres": 0.45,
                "debug_image": True,
            }],
            output="screen",
        ),
    ])
