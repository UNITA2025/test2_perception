from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import GroupAction

def generate_launch_description():
    ld = LaunchDescription()

    camera_names = ['front_up', 'front_down', 'left', 'right']
    yolo_nodes = []

    for name in camera_names:
        image_topic = f'/camera_{name}/image_raw'
        detection_topic = f'/detections/{name}'

        # 카메라별 모델 선택
        if name == 'front_up':
            continue   # ✅ front_up 전용 모델
        else:
            model_file = 'best.pt'       # ✅ 나머지 카메라 공통 모델

        yolo_node = Node(
            package='perception_yolov8_pkg',
            executable='yolov8_node',
            name=f'yolov8_{name}',
            output='screen',
            emulate_tty=True,
            parameters=[
                {'model': model_file},              # ✅ 카메라마다 다른 모델 적용
                {'device': 'cuda:0'},
                {'threshold': 0.5},
                {'enable': True},
                {'image_reliability': 1},
                {'image_topic': image_topic},
            ],
            remappings=[
                ('detections', detection_topic)
            ]
        )
        yolo_nodes.append(yolo_node)

    ld.add_action(GroupAction(yolo_nodes))
    return ld