#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
from interfaces_pkg.msg import PathPlanningResult, ConeInfoArray
#from interfaces_pkg.msg import ErpCmdMsg1

from interfaces_control_pkg.msg import ErpCmdMsg


# -------------------------------
# Topic 이름 설정
SUB_PATH_TOPIC = '/path_planning_result'      # 경로 계획 결과 토픽
SUB_CONE_TOPIC = '/drums/drum_info_down'      # 콘 감지 토픽
PUB_ERP_CMD   = '/erp42_ctrl_cmd'             # 차량 제어 퍼블리시 토픽

# -------------------------------
# 차량 제어 관련 하이퍼파라미터
DEFAULT_SPEED_CMD   = 0      # 기본 속도 (0~200)
STEER_LIMIT_DEG     = 30.0    # 최대 조향각 (degree)
STEER_LIMIT_CNT     = 2000    # 최대 스티어 카운트 (하드웨어 매핑용)
LOOKAHEAD_POINT_IDX = 10      # 목표 기울기 계산 시 참조할 이전 경로 점 개수

# -------------------------------
# 타이머 관련
TIMER_PERIOD        = 0.05    # 모션 명령 퍼블리시 주기 (초)

# -------------------------------
# 장애물 회피 관련
OBSTACLE_X_LIMIT    = 1.0     # 전방 장애물 감지 좌/우 범위 (m)
OBSTACLE_Y_MIN      = 0       # 전방 장애물 최소 거리 (m)
OBSTACLE_Y_MAX      = 2.0     # 전방 장애물 최대 거리 (m)

# -------------------------------
# QoS 설정
QOS_DEPTH           = 5

# -------------------------------
class PathConeMotionPlanner(Node):
    def __init__(self):
        super().__init__('erp_motion_planner_node')

        # QoS
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=QOS_DEPTH
        )

        # Subscribers
        self.path_sub = self.create_subscription(PathPlanningResult, SUB_PATH_TOPIC, self.path_callback, qos)
        self.cone_sub = self.create_subscription(ConeInfoArray, SUB_CONE_TOPIC, self.cone_callback, qos)

        # Publisher
        self.erp_pub = self.create_publisher(ErpCmdMsg, PUB_ERP_CMD, qos)

        # 변수 초기화
        self.path_points = None
        self.cones = None
        self.last_steer_cnt = 0
        self.last_speed_cmd = DEFAULT_SPEED_CMD

        # 주기적으로 제어 명령 퍼블리시
        self.timer = self.create_timer(TIMER_PERIOD, self.timer_callback)

    def path_callback(self, msg: PathPlanningResult):
        self.path_points = list(zip(msg.x_points, msg.y_points))

    def cone_callback(self, msg: ConeInfoArray):
        self.cones = msg.cones

    def timer_callback(self):
        # 차량 제어 기본값
        steer_deg = 0.0
        speed_cmd = self.last_speed_cmd

        # --- 경로 기반 조향 계산 ---
        if self.path_points and len(self.path_points) > LOOKAHEAD_POINT_IDX:
            # 마지막 lookahead_point_idx 번째 포인트와 끝점으로 목표 기울기 계산
            start_pt = self.path_points[-LOOKAHEAD_POINT_IDX]
            end_pt = self.path_points[-1]
            dx = end_pt[0] - start_pt[0]
            dy = end_pt[1] - start_pt[1]
            if dy != 0:
                slope_deg = math.degrees(math.atan2(dx, dy))
                # 조향각 제한 적용
                steer_deg = max(min(slope_deg, STEER_LIMIT_DEG), -STEER_LIMIT_DEG)

        # --- 콘 정보로 장애물 회피 간단 처리 ---
        if self.cones:
            for c in self.cones:
                if OBSTACLE_Y_MIN < c.y < OBSTACLE_Y_MAX and abs(c.x) < OBSTACLE_X_LIMIT:
                    speed_cmd = 0
                    self.get_logger().info(f"Obstacle detected at x:{c.x:.2f}, y:{c.y:.2f} -> Stop")
                    break

        # --- 조향각 변환 (deg -> steer count) ---
        steer_cnt = int(steer_deg / STEER_LIMIT_DEG * STEER_LIMIT_CNT)
        steer_cnt = max(min(steer_cnt, STEER_LIMIT_CNT), -STEER_LIMIT_CNT)

        self.last_steer_cnt = steer_cnt
        self.last_speed_cmd = speed_cmd

        # --- ErpCmdMsg 생성 및 퍼블리시 ---
        cmd_msg = ErpCmdMsg()
        cmd_msg.steer = steer_cnt
        cmd_msg.speed = speed_cmd
        cmd_msg.gear = 0
        cmd_msg.brake = 0
        cmd_msg.e_stop = False
        self.erp_pub.publish(cmd_msg)

        self.get_logger().info(f"Steer: {steer_deg:.2f} deg / cnt:{steer_cnt}, Speed:{speed_cmd}")


def main(args=None):
    rclpy.init(args=args)
    node = PathConeMotionPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("shutdown!")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
