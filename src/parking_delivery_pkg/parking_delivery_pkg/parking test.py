#!/usr/bin/env python3
import math
import time
from collections import deque

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from interfaces_pkg.msg import ErpCmdMsg1, ErpStatusMsg1
from interfaces_pkg.msg import ConeInfoArray

class DiagonalParking(Node):
    def __init__(self):
        super().__init__('diagonal_parking')

        # --- 퍼블리셔 & 구독자 ---
        self.erp_cmd_pub = self.create_publisher(ErpCmdMsg1, '/erp42_ctrl_cmd', 10)
        self.status_sub = self.create_subscription(ErpStatusMsg1, '/erp42_status', self.status_callback, 10)
        self.cone_sub = self.create_subscription(ConeInfoArray, '/cones/cone_info_down', self.cone_callback, 10)

        # --- 상태 변수 ---
        self.current_speed = 0.0
        self.current_steering_angle = 0.0
        self.lat_err = deque(maxlen=100)
        self.PID_steer = self.PID(self)

        self.mode = "IDLE"  # IDLE, PARKING, PARKED, EXIT
        self.parking_waypoints = []  # 사선주차 진입용 waypoint
        self.exit_waypoints = []     # 출차용 waypoint
        self.waypoint_index = 0

        self.vehicle_position = [0.0, 0.0]  # GNSS 대체 변수
        self.wheel_base = 1.04

        # 주행/제어 루프 10Hz
        self.control_timer = self.create_timer(0.1, self.control_loop)

        # --- 테스트용: 1초 후 자동 PARKING 모드 진입 ---
        self.create_timer(1.0, self.force_parking_test)

    # --- 테스트용 함수 ---
    def force_parking_test(self):
        if self.mode == "IDLE":
            self.parking_waypoints = [
                Point(x=1.0, y=0.0, z=0.0),
                Point(x=2.0, y=-0.5, z=0.0)
            ]
            self.exit_waypoints = [
                Point(x=0.0, y=0.0, z=0.0)
            ]
            self.waypoint_index = 0
            self.mode = "PARKING"
            self.get_logger().info("테스트용: PARKING 모드 강제 전환")

    # --- 차량 상태 수신 ---
    def status_callback(self, msg: ErpStatusMsg1):
        self.current_speed = msg.speed
        self.current_steering_angle = (msg.steer / 2000.0) * 30.0

    # --- 콘 정보 수신 ---
    def cone_callback(self, msg: ConeInfoArray):
        if self.mode != "IDLE":
            return  # 주차 중에는 무시

        left_cones, right_cones = [], []
        for c in msg.cones:
            if c.cone_color == "blue":
                left_cones.append(c)
            elif c.cone_color == "yellow":
                right_cones.append(c)

        for l, r in zip(left_cones, right_cones):
            dist = math.hypot(r.x - l.x, r.y - l.y)
            if dist > 1.8:
                self.parking_waypoints = [
                    Point(x=(l.x + r.x)/2, y=(l.y + r.y)/2, z=0.0),
                    Point(x=(l.x + r.x)/2 + 1.0, y=(l.y + r.y)/2 - 0.5, z=0.0)
                ]
                self.exit_waypoints = [
                    Point(x=self.vehicle_position[0] - 1.0, y=self.vehicle_position[1] + 1.0, z=0.0)
                ]
                self.waypoint_index = 0
                self.mode = "PARKING"
                self.get_logger().info("빈 주차 공간 발견! PARKING 모드로 전환")
                break

    # --- 제어 루프 ---
    def control_loop(self):
        if self.mode == "IDLE":
            self.publish_cmd(steer=0, speed=40)
            return

        if self.mode in ["PARKING", "EXIT"]:
            waypoints = self.parking_waypoints if self.mode == "PARKING" else self.exit_waypoints
            if self.waypoint_index < len(waypoints):
                target_wp = waypoints[self.waypoint_index]
                dx = target_wp.x - self.vehicle_position[0]
                dy = target_wp.y - self.vehicle_position[1]
                dist = math.hypot(dx, dy)

                if dist < 0.2:
                    self.waypoint_index += 1
                    if self.mode == "PARKING" and self.waypoint_index >= len(self.parking_waypoints):
                        self.mode = "PARKED"
                        self.get_logger().info("주차 완료! PARKED 모드로 전환")
                    elif self.mode == "EXIT" and self.waypoint_index >= len(self.exit_waypoints):
                        self.mode = "IDLE"
                        self.get_logger().info("출차 완료! IDLE 모드로 전환")
                    return

                # Pure Pursuit Steering
                angle_to_wp = math.atan2(dy, dx)
                delta_deg = math.degrees(math.atan2(2.0 * self.wheel_base * math.sin(angle_to_wp), max(dist, 1e-3)))
                delta_err = delta_deg - self.current_steering_angle
                steer_cmd = self.PID_steer.control(delta_err)
                steer_cmd = max(min(steer_cmd, 30.0), -30.0)
                steering_angle_cmd = int((steer_cmd / 30.0) * 2000)

                speed = 15 if self.mode == "PARKING" else 40
                self.publish_cmd(steer=steering_angle_cmd, speed=speed)

        elif self.mode == "PARKED":
            self.publish_cmd(steer=0, speed=0)
            time.sleep(5)
            self.mode = "EXIT"
            self.waypoint_index = 0
            self.get_logger().info("자동 출차! EXIT 모드로 전환")

    # --- ERP42 제어 메시지 퍼블리시 ---
    def publish_cmd(self, steer, speed):
        msg = ErpCmdMsg1()
        msg.steer = steer
        msg.speed = speed
        msg.gear = 0
        msg.e_stop = False
        msg.brake = 0
        self.erp_cmd_pub.publish(msg)

    # --- PID ---
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
            return self.Pterm + self.ki * self.Iterm + self.Dterm


def main(args=None):
    rclpy.init(args=args)
    node = DiagonalParking()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
