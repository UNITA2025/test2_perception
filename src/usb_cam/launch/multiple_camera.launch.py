import os
from pathlib import Path
import sys

# 상대 import용 패스 설정
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

from camera_config import CameraConfig, USB_CAM_DIR

from launch import LaunchDescription
from launch.actions import GroupAction
from launch_ros.actions import Node

# 여기서 4개 카메라를 선언
CAMERAS = [
    CameraConfig(
        name='camera_front_up',
        param_path=Path(USB_CAM_DIR, 'config', 'params_1_front_up.yaml')
    ),
    CameraConfig(
        name='camera_front_down',
        param_path=Path(USB_CAM_DIR, 'config', 'params_2_front_down.yaml')
    ),
    CameraConfig(
        name='camera_left',
        param_path=Path(USB_CAM_DIR, 'config', 'params_3_left.yaml')
    ),
    CameraConfig(
        name='camera_right',
        param_path=Path(USB_CAM_DIR, 'config', 'params_4_right.yaml')
    )
]
# LaunchDescription 생성 및 카메라 노드 설정

def generate_launch_description():
    ld = LaunchDescription()

    camera_nodes = [
        Node(
            package='usb_cam',
            executable='usb_cam_node_exe',
            output='screen',
            name=camera.name,
            namespace=camera.namespace,
            parameters=[camera.param_path],
            remappings=camera.remappings
        )
        for camera in CAMERAS
    ]

    camera_group = GroupAction(camera_nodes)
    ld.add_action(camera_group)

    return ld
