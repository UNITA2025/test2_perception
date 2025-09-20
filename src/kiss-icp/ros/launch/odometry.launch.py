# import os

# from ament_index_python.packages import get_package_share_directory
# from launch import LaunchDescription
# from launch.actions import DeclareLaunchArgument, ExecuteProcess
# from launch.conditions import IfCondition
# from launch.substitutions import (
#     LaunchConfiguration,
#     PathJoinSubstitution,
# )
# from launch_ros.actions import Node
# from launch_ros.substitutions import FindPackageShare

# PACKAGE_NAME = "kiss_icp"

# default_config_file = os.path.join(
#     get_package_share_directory(PACKAGE_NAME), "config", "config.yaml"
# )

# def generate_launch_description():
#     # -------- Launch arguments --------
#     use_sim_time     = LaunchConfiguration("use_sim_time")       # 실차: false / bag: true
#     visualize        = LaunchConfiguration("visualize")          # RViz 띄울지
#     pointcloud_topic = LaunchConfiguration("topic")              # 라이다 토픽
#     bagfile          = LaunchConfiguration("bagfile")            # bag 경로 (옵션)
#     start_bag        = LaunchConfiguration("start_bag")          # bag 재생 여부 (옵션)

#     base_frame       = LaunchConfiguration("base_frame")
#     lidar_odom_frame = LaunchConfiguration("lidar_odom_frame")
#     publish_odom_tf  = LaunchConfiguration("publish_odom_tf")
#     invert_odom_tf   = LaunchConfiguration("invert_odom_tf")
#     position_cov     = LaunchConfiguration("position_covariance")
#     orientation_cov  = LaunchConfiguration("orientation_covariance")
#     config_file      = LaunchConfiguration("config_file")

#     declare_args = [
#         DeclareLaunchArgument("use_sim_time",      default_value="false"),       # 실차 기본: false
#         DeclareLaunchArgument("visualize",         default_value="true"),
#         DeclareLaunchArgument("topic",             default_value="velodyne_points"),
#         DeclareLaunchArgument("bagfile",           default_value=""),            # 공백이면 미재생
#         DeclareLaunchArgument("start_bag",         default_value="false"),       # 필요할 때만 true

#         DeclareLaunchArgument("base_frame",        default_value="base_link"),
#         DeclareLaunchArgument("lidar_odom_frame",  default_value="odom"),
#         DeclareLaunchArgument("publish_odom_tf",   default_value="true"),
#         DeclareLaunchArgument("invert_odom_tf",    default_value="false"),
#         DeclareLaunchArgument("position_covariance",    default_value="0.1"),
#         DeclareLaunchArgument("orientation_covariance", default_value="0.1"),
#         DeclareLaunchArgument("config_file",       default_value=default_config_file),
#     ]

#     # -------- Nodes --------
#     kiss_icp_node = Node(
#         package=PACKAGE_NAME,
#         executable="kiss_icp_node",
#         name="kiss_icp_node",
#         output="screen",
#         remappings=[("pointcloud_topic", pointcloud_topic)],
#         parameters=[{
#             "base_frame": base_frame,
#             "lidar_odom_frame": lidar_odom_frame,
#             "publish_odom_tf": publish_odom_tf,
#             "invert_odom_tf": invert_odom_tf,
#             "publish_debug_clouds": visualize,
#             "use_sim_time": use_sim_time,
#             "position_covariance": position_cov,
#             "orientation_covariance": orientation_cov,
#         }, config_file],
#     )

#     rviz_node = Node(
#         package="rviz2",
#         executable="rviz2",
#         output="screen",
#         arguments=[
#             "-d",
#             PathJoinSubstitution([FindPackageShare(PACKAGE_NAME), "rviz", "kiss_icp.rviz"]),
#         ],
#         parameters=[{"use_sim_time": use_sim_time}],
#         condition=IfCondition(visualize),
#     )

#     # bag 재생은 옵션 (start_bag=true & bagfile != "")
#     bagfile_play = ExecuteProcess(
#         cmd=["ros2", "bag", "play", bagfile, "--rate", "1", "--clock"],
#         output="screen",
#         condition=IfCondition(start_bag),
#     )

#     # 실차/백 공통: 라이다→베이스 링크 정적 TF
#     static_tf = Node(
#         package="tf2_ros",
#         executable="static_transform_publisher",
#         arguments=[
#             "--x","0.614","--y","0.0","--z","-0.505",
#             "--roll","0.0","--pitch","0.0","--yaw","0.0",
#             "--frame-id","base_link","--child-frame-id","velodyne",
#         ],
#         output="screen",
#     )

#     return LaunchDescription(
#         declare_args + [
#             #   static_tf,         # 보통 먼저 올라가 있어도 무방
#             kiss_icp_node,
#             rviz_node,
#             #bagfile_play,      # 필요할 때만 실행
#         ]
#     )


# MIT License
#
# Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
# Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
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
    bagfile = LaunchConfiguration("bagfile", default="")

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
        condition=IfCondition(visualize),
    )

    bagfile_play = ExecuteProcess(
        cmd=["ros2", "bag", "play", "--rate", "1", bagfile, "--clock", "1000.0"],
        output="screen",
        condition=IfCondition(PythonExpression(["'", bagfile, "' != ''"])),
    )

    return LaunchDescription(
        [
            kiss_icp_node,
            rviz_node,
            bagfile_play,
        ]
    )