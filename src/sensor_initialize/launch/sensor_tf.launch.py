from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 기존: LiDAR-카메라 TF 브로드캐스터
        Node(
            package='sensor_initialize',
            executable='lidar_to_camera_broadcaster',
            name='lidar_camera_tf',
            parameters=[{
                'config_path': '/home/unita/test2_perception/src/sensor_initialize/config/'
            }]
        ),

        # 추가: base_link → velodyne 정적 TF (필요시 값 수정)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_velodyne',
            # x y z yaw pitch roll parent child
            arguments=['0.614', '0', '0.7', '0', '0', '0', 'base_link', 'velodyne']
        ),
    ])
