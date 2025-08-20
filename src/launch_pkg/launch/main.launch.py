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
    info_pkg_dir = get_package_share_directory('info_publisher_pkg')
    lane_tracking_pkg_dir = get_package_share_directory('lane_tracking_pkg')

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

        # 2) YOLOv8 추론
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(yolov8_dir, 'launch', 'multi_camera_yolov8.launch.py')
            )
        ),

        # 3) 디버그 시각화
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(debug_dir, 'launch', 'multi_camera_yolov8_debug.launch.py')
            )
        ),

        # 4) TF 브로드캐스트 (카메라 ⇆ LiDAR)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(sensor_init_dir, 'launch', 'sensor_tf.launch.py')
            )
        ),

        # 5) LiDAR 전처리
        Node(
           package='sensor_initialize',
           executable='lidar_preprocessing_node',
           name='lidar_preprocessor',
           output='screen',
           parameters=[{
                # 필요하면 파라미터 여기
           }]
        ),

        # # 6) LiDAR-카메라 Fusion (멀티 실행)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(fusion_pkg_dir, 'launch', 'fusion_multi.launch.py')
            )
        ),

        # 7) msg 발행?!? (cone, drum)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(info_pkg_dir, 'launch', 'info_publisher.launch.py')
            )
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(lane_tracking_pkg_dir, 'launch', 'lane_tracking.launch.py')
            )
        ),
    ])

