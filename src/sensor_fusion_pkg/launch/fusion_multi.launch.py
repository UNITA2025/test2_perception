from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Left LiDAR Fusion
        Node(
            package='sensor_fusion_pkg',
            executable='bbox_projector_node',
            name='fusion_left',
            output='screen',
            parameters=[
                {'det_topic': '/detections/left'},
                {'lidar_topic': '/lidar_pre_left'},
                {'camera_frame': 'camera_left_frame'},
                {'camera_image_topic': '/camera_left/image_raw'},
                {'cloud_out_cone': '/cones_left/colored_points_cone'},
                {'cloud_out_drum': '/cones_left/colored_points_drum'},
                {'marker_out': '/cones_left/markers'},
                {'projection_matrix': [520.0, 0.0, 310.0, 0.0,
                                       0.0, 525.0, 245.0, 0.0,
                                       0.0, 0.0, 1.0, 0.0]},
            ]
        ),
        # Right LiDAR Fusion
        Node(
            package='sensor_fusion_pkg',
            executable='bbox_projector_node',
            name='fusion_right',
            output='screen',
            parameters=[
                {'det_topic': '/detections/right'},
                {'lidar_topic': '/lidar_pre_right'},
                {'camera_frame': 'camera_right_frame'},
                {'camera_image_topic': '/camera_right/image_raw'},
                {'cloud_out_cone': '/cones_right/colored_points_cone'},
                {'cloud_out_drum': '/cones_right/colored_points_drum'},
                {'marker_out': '/cones_right/markers'},
                {'projection_matrix': [530.0, 0.0, 318.0, 0.0,
                                       0.0, 528.0, 250.0, 0.0,
                                       0.0, 0.0, 1.0, 0.0]},
            ]
        ),
        # Down LiDAR Fusion
        Node(
            package='sensor_fusion_pkg',
            executable='bbox_projector_node',
            name='fusion_down',
            output='screen',
            parameters=[
                {'det_topic': '/detections/front_down'},
                {'lidar_topic': '/lidar_pre_down'},
                {'camera_frame': 'camera_front_down_frame'},
                {'camera_image_topic': '/camera_front_down/image_raw'},
                {'cloud_out_cone': '/cones_down/colored_points_cone'},
                {'cloud_out_drum': '/cones_down/colored_points_drum'},
                {'marker_out': '/cones_down/markers'},
                {'projection_matrix': [533.32761, 0.0, 316.61534, 0.0,
                                       0.0, 533.25104, 248.71553, 0.0,
                                       0.0, 0.0, 1.0, 0.0]},
            ]
        ),
    ])
