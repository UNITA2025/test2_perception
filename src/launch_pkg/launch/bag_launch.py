from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # 패키지 경로
    yolov8_dir       = get_package_share_directory('perception_yolov8_pkg')
    debug_dir        = get_package_share_directory('debug_pkg')
    sensor_init_dir  = get_package_share_directory('sensor_initialize')
    fusion_pkg_dir   = get_package_share_directory('sensor_fusion_pkg')
    info_pkg_dir = get_package_share_directory('info_publisher_pkg')


    return LaunchDescription([
        # 1) YOLOv8 추론
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(yolov8_dir, 'launch', 'multi_camera_yolov8.launch.py')
            )
        ),

        # 2) 디버그 시각화
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(debug_dir, 'launch', 'multi_camera_yolov8_debug.launch.py')
            )
        ),

        # 3) TF 브로드캐스트 (카메라 ⇆ LiDAR)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(sensor_init_dir, 'launch', 'sensor_tf.launch.py')
            )
        ),

        # 4) LiDAR 전처리
        Node(
           package='sensor_initialize',
           executable='lidar_preprocessing_node',
           name='lidar_preprocessor',
           output='screen',
           parameters=[{
                # 필요하면 파라미터 여기
           }]
        ),

        # 5) LiDAR-카메라 Fusion (멀티 실행)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(fusion_pkg_dir, 'launch', 'fusion_multi.launch.py')
            )
        ),

        # 6) msg 발행?!? (cone, drum)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(info_pkg_dir, 'launch', 'info_publisher.launch.py')
            )
        ),
    ])

