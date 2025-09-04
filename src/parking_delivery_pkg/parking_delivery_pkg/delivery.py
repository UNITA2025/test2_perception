#!/usr/bin/env python3
import math
import time
from collections import deque

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

from interfaces_control_pkg.msg import ErpCmdMsg, ErpStatusMsg
from interfaces_pkg.msg import VisionSignArray  # 카메라 표지판 메시지

class DeliveryMission(Node):
    def __init__(self):
        super().__init__('delivery_mission')

        # ERP42 제어 메시지
        self.ctrl_cmd_msg = ErpCmdMsg()
        self.erp_cmd_pub = self.create_publisher(ErpCmdMsg, '/erp42_ctrl_cmd', 10)

        # 구독
        self.status_sub = self.create_subscription(ErpStatusMsg, '/erp42_status', self.status_callback, 10)
        self.path_sub = self.create_subscription(Path, '/waypoint_path', self.path_callback, 10)
        self.sign_sub = self.create_subscription(VisionSignArray, '/vision_sign', self.sign_callback, 10)

        # 상태 변수
        self.current_speed = 0
        self.current_steering_angle = 0
        self.waypoints = []
        self.waypoint_index = 0
        self.lat_err = deque(maxlen=100)
        self.PID_steer = self.PID(self)

        # 배달 미션 상태
        self.delivery_stage = 'A'  # 'A', 'B', 'DONE'
        self.delivery_target_sign = None
        self.delivery_stop_time = 3.0  # 정차 시간 (초)
        self.delivery_area_gnss = {  # 예시 GNSS 영역 (x_min, x_max, y_min, y_max)
            'A': [10, 20, -5, 5],
            'B': [50, 60, -5, 5]
        }

        # 10Hz 제어 루프
        self.control_timer = self.create_timer(0.1, self.control_loop)

    # ERP42 상태 콜백
    def status_callback(self, msg: ErpStatusMsg):
        self.current_speed = msg.speed
        self.current_steering_angle = (msg.steer / 2000.0) * 30.0

    # 경로 콜백
    def path_callback(self, msg: Path):
        self.waypoints = [[ps.pose.position.x, ps.pose.position.y] for ps in msg.poses]
        self.waypoint_index = 0

    # 표지판 콜백
    def sign_callback(self, msg: VisionSignArray):
        for sign in msg.signs:
            if sign.label.startswith('A') and self.delivery_stage == 'A':
                self.delivery_target_sign = sign
            elif sign.label.startswith('B') and self.delivery_stage == 'B':
                self.delivery_target_sign = sign

    # GNSS 영역 판단 (현재 위치가 배달 구역 내인지)
    def in_delivery_area(self, stage, x, y):
        if stage not in self.delivery_area_gnss:
            return False
        x_min, x_max, y_min, y_max = self.delivery_area_gnss[stage]
        return x_min <= x <= x_max and y_min <= y <= y_max

    # 제어 루프
    def control_loop(self):
        if not self.waypoints or self.waypoint_index >= len(self.waypoints):
            return

        # 현재 목표 웨이포인트
        wx, wy = self.waypoints[self.waypoint_index]
        dx = wx - 0  # 현재 x 위치 (GNSS 적용 시 수정)
        dy = wy - 0  # 현재 y 위치 (GNSS 적용 시 수정)
        dist = math.hypot(dx, dy)

        # 배달 구역 진입 감지 (예시 GNSS 좌표 사용)
        if self.delivery_stage in ['A', 'B'] and self.in_delivery_area(self.delivery_stage, 0, 0):
            self.get_logger().info(f"Entered delivery {self.delivery_stage} area. Slowing down.")
            speed = 20  # 속도 감소

            if self.delivery_target_sign:
                # 표지판 감지 시 정차
                self.get_logger().info(f"Detected delivery sign {self.delivery_target_sign.label}. Stopping.")
                self.stop_vehicle(self.delivery_stop_time)
                # 배달 후 stage 업데이트
                if self.delivery_stage == 'A':
                    self.delivery_stage = 'B'
                elif self.delivery_stage == 'B':
                    self.delivery_stage = 'DONE'
                self.delivery_target_sign = None
                return
        else:
            speed = 40  # 일반 주행 속도

        # 웨이포인트 도달 체크
        if dist < 0.5:
            self.waypoint_index += 1
            return

        # pure-pursuit 간단 구현
        angle_to_wp = math.atan2(dy, dx)
        angle_error = (angle_to_wp + math.pi) % (2*math.pi) - math.pi
        wheel_base = 1.04
        delta_deg = math.degrees(math.atan2(2.0 * wheel_base * math.sin(angle_error), dist))
        steering_angle_deg = max(min(delta_deg, 30.0), -30.0)

        # PID 보정
        delta_err = steering_angle_deg - self.current_steering_angle
        steer_pid = self.PID_steer.control(delta_err)
        steer_pid = max(min(steer_pid, 30.0), -30.0)
        steering_angle_cmd = int((steer_pid / 30.0) * 2000)
        steering_angle_cmd = max(min(steering_angle_cmd, 2000), -2000)

        # ERP42 제어 퍼블리시
        self.ctrl_cmd_msg.steer = steering_angle_cmd
        self.ctrl_cmd_msg.speed = speed
        self.ctrl_cmd_msg.gear = 0
        self.ctrl_cmd_msg.e_stop = False
        self.ctrl_cmd_msg.brake = 0
        self.erp_cmd_pub.publish(self.ctrl_cmd_msg)

    # 차량 정지
    def stop_vehicle(self, duration):
        end_time = time.time() + duration
        while time.time() < end_time and rclpy.ok():
            self.ctrl_cmd_msg.steer = 0
            self.ctrl_cmd_msg.speed = 0
            self.ctrl_cmd_msg.gear = 0
            self.ctrl_cmd_msg.e_stop = False
            self.ctrl_cmd_msg.brake = 30
            self.erp_cmd_pub.publish(self.ctrl_cmd_msg)
            time.sleep(0.1)

    # PID 클래스
    class PID:
        def __init__(self, outer):
            self.node = outer
            self.kp = 1.0
            self.ki = 0.001
            self.kd = 0.001
            self.Pterm = 0.0
            self.Iterm = 0.0
            self.Dterm = 0.0
            self.prev_error = 0.0
            self.dt = 0.1

        def control(self, error):
            self.Pterm = self.kp * error
            self.Dterm = self.kd * (error - self.prev_error) / self.dt
            if abs(error) < 15.0:
                self.Iterm += error * self.dt
            self.prev_error = error
            output = self.Pterm + self.ki * self.Iterm + self.Dterm
            return output

def main(args=None):
    rclpy.init(args=args)
    node = DeliveryMission()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
    
#delivery_stage 상태 관리 ('A' → 'B' → 'DONE')
#GNSS 기반 배달 구역 감지 (in_delivery_area)
#카메라 표지판 감지 (VisionSignArray) → 배달 정차 위치 결정
#정차 후 일정 시간 정지 후 기존 경로로 복귀
#ERP42 제어 메시지 타입 통일 (ErpCmdMsg, ErpStatusMsg)
#간단한 pure-pursuit로 경로 따라 주행
