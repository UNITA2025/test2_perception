import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

PACKAGE_NAME = "kiss_icp"

default_config_file = os.path.join(
    get_package_share_directory(PACKAGE_NAME), "config", "config.yaml"
)


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time", default="true")

    # ROS configuration
    pointcloud_topic = LaunchConfiguration("topic", default="/velodyne_points")
    visualize = LaunchConfiguration("visualize", default="true")

    # Optional ros bag play
    bagfile = LaunchConfiguration("bagfile", default="0907_5")

    # tf tree configuration, these are the likely parameters to change and nothing else
    base_frame = LaunchConfiguration("base_frame", default="base_link")  # (base_link/base_footprint)
    lidar_odom_frame = LaunchConfiguration("lidar_odom_frame", default="odom")
    publish_odom_tf = LaunchConfiguration("publish_odom_tf", default=True)
    invert_odom_tf = LaunchConfiguration("invert_odom_tf", default=False)

    position_covariance = LaunchConfiguration("position_covariance", default=0.1)
    orientation_covariance = LaunchConfiguration("orientation_covariance", default=0.1)

    config_file = LaunchConfiguration("config_file", default=default_config_file)

    # KISS-ICP node
    kiss_icp_node = Node(
        package=PACKAGE_NAME,
        executable="kiss_icp_node",
        name="kiss_icp_node",
        output="screen",
        remappings=[
            ("pointcloud_topic", pointcloud_topic),
        ],
        parameters=[
            {
                # ROS node configuration
                "base_frame": base_frame,
                "lidar_odom_frame": lidar_odom_frame,
                "publish_odom_tf": publish_odom_tf,
                "invert_odom_tf": invert_odom_tf,
                # ROS CLI arguments
                "publish_debug_clouds": visualize,
                "use_sim_time": use_sim_time,
                "position_covariance": position_covariance,
                "orientation_covariance": orientation_covariance,
            },
            config_file,
        ],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        output="screen",
        arguments=[
            "-d",
            PathJoinSubstitution([FindPackageShare(PACKAGE_NAME), "rviz", "kiss_icp.rviz"]),
        ],
        parameters=[
            {
                'use_sim_time': use_sim_time
            },
        ],
        condition=IfCondition(visualize),
    )

    bagfile_play = ExecuteProcess(
        cmd=["ros2", "bag", "play", bagfile, "--rate", "1", "--clock"],
        output="screen",
        condition=IfCondition(PythonExpression(["'", bagfile, "' != ''"])),
    )

    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[
            "--x","0.614","--y","0.0","--z","-0.505",
            "--roll","0.0","--pitch","0.0","--yaw","0.0",
            "--frame-id","base_link","--child-frame-id","velodyne",
        ],
        output="screen",
    )


    return LaunchDescription(
        [
            kiss_icp_node,
            rviz_node,
            # bagfile_play,
            static_tf,
        ]
    )
