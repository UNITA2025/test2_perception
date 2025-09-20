from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node, SetParameter
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration

def generate_launch_description():
    # CLI에서 덮어쓸 수 있게 인자화 (기본 true)
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    kiss_icp_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                get_package_share_directory('kiss_icp'),
                'launch', 'odometry.launch.py'
            ])
        ),
        # odometry.launch.py 쪽에도 명시적으로 전달 (네 파일 기본값이 true여도 명시 전달 추천)
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    node_test = Node(
        package='test_pkg', executable='test', name='test',
        parameters=[{'use_sim_time': use_sim_time}]
    )
    node_vis  = Node(
        package='test_pkg', executable='test_visualizer', name='test_visualizer',
        parameters=[{'use_sim_time': use_sim_time}]
    )
    node_plan = Node(
        package='test_pkg', executable='test_planner', name='test_planner',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    node_fusion = Node(
        package='test_pkg', executable='fusion_buffer_node', name='fusion_buffer_node',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    node_odom2tf = Node(
        package='test_pkg', executable='odom2tf', name='odom2tf',
        parameters=[{
            'use_sim_time': use_sim_time,
            'odom_topic': '/kiss/odometry',
            'parent_frame': 'odom',
            'child_frame': 'base_link',
            }],
    )

    return LaunchDescription([
        # 전역 인자 (ros2 launch ... use_sim_time:=false 로 끌 수 있음)
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        # 이 줄 하나로 런치 트리 아래 모든 노드에 use_sim_time 전파
        SetParameter(name='use_sim_time', value=use_sim_time),

        kiss_icp_launch,
        #node_odom2tf,
        
        TimerAction(period=2.0, actions=[node_test, node_fusion,]),
    ])
