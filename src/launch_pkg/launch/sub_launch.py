from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # 패키지 경로
    usb_cam_dir      = get_package_share_directory('usb_cam')
    yolov8_dir       = get_package_share_directory('perception_yolov8_pkg')
    debug_dir        = get_package_share_directory('debug_pkg')
    sensor_init_dir  = get_package_share_directory('sensor_initialize')
    velodyne_driver_dir  = get_package_share_directory('velodyne_driver')
    velodyne_pointcloud_dir  = get_package_share_directory('velodyne_pointcloud')
    fusion_pkg_dir   = get_package_share_directory('sensor_fusion_pkg')  # Fusion 패키지 경로
    # lidar_pre_dir    = get_package_share_directory('lidar_preprocessing_pkg')  # 전처리 패키지
    # fusion_pkg_dir   = get_package_share_directory('sensor_fusion_pkg')

    return LaunchDescription([

        # 0) 라이다 드라이버 퍼블리셔
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(velodyne_driver_dir, 'launch', 'velodyne_driver_node-VLP16-launch.py')
            )
        ),
        # 0) 라이다 포인트클라우드 퍼블리셔
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(velodyne_pointcloud_dir, 'launch', 'velodyne_transform_node-VLP16-launch.py')
            )
        ),

        # 1) 카메라 퍼블리셔
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(usb_cam_dir, 'launch', 'multiple_camera.launch.py')
            )
        ),
    ])

