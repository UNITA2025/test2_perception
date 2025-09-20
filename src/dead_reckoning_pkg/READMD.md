# dead_reckoning_pkg

EPR-42 기반 자율주행 프로젝트용 ROS2 패키지
GPS를 사용하지 못하는 음영구간 내에서 LIDAR만을 사용하여 로컬 주행 및 시물레이션을 지원합니다.

---

## 패키지 구성 

- **pointcloud_filter_node**
    Velodnye LIDAR Data preprocees -> '/lane_points', '/drum_points' publish 

- **pointcloud_filter_visualizer_node**
    전처리된 포인터들을 Rviz2에서 시각화 (파랑: lane, 빨간: drum)

- **mapping_node**
    kiss_icp 기반 mapping -> 'lane_map', 'drum_map' publish

- **lane_planner_node**
    생성된 맵을 기반으로 주행 경로 생성 -> '/center_path' publish

- **path_follower_node**
    EPR-42 제어 명령 생성 -> '/erp42_ctrl_cmd' publish

## 패키지 사용법

### 정적 tf 생성
ros2 run tf2_ros static_transform_publisher \
--x 0.614 --y 0.0 --z -0.505 \
--roll 0.0 --pitch 0.0 --yaw 0.0 \
--frame-id base_link \
--child-frame-id velodyne

### bag 실행 or velodyne 실행 (아래 코드 중 상황에 맞게 사용하시면 됩니다.)
#### 주의 
    bag을 사용할때는 kiss_icp 안에 launch와 dead_reckoning_pkg 안에 launch에 use_sim_time 값을 "true"
    velodyne을 사용할떄는 use_sim_time 값을 "false"로 설정해주세요

    ros2 bag play {bag_name} --clock 1000 --rate 1 
    (clock: 시간 발행 -> bag에 저장된 시간이 없다면 임의로 시간을 내보내줘야 tf를 잡을 수 있습니다.
    rate: bag 속도 -> 백이 느려진다면 rate를 0.5에서 1.0 사이로 조절하시면 됩니다.)



    ros2 launch launch_pkg sub_launch.py

### launch 실행
#### 참고
    rviz2가 필요하다면 kiss_icp 안에 launch에서 "visualize"값을 'true' 
    rviz2가 필요없다면 kiss_icp 안에 launch에서 "visualize"값을 'false' 

    ros2 launch dead_reckoning_pkg dead_reckoning.launch.py

