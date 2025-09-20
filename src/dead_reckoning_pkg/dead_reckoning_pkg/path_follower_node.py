#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
)

from nav_msgs.msg import Path
from interfaces_control_pkg.msg import ErpCmdMsg

from interfaces_pkg.msg import ConeInfoArray  # (옵션) 장애물 회피용
from simple_pid import PID   ### [PID 추가]

## 차량이 중앙 경로(Path)를 따라가도록 조향 및 속도를 계산해서 제어 명령을 퍼블리시 ##

class PathFollower(Node):
    def __init__(self):
        super().__init__('path_follower_node')

        # ===== 파라미터 =====
        # 입력 경로 Path 토픽 (보통 x=전방, y=좌우)
        self.declare_parameter('center_path_topic', '/center_path')
        self.declare_parameter('center_swap_xy', True)   # True면 (x,y)->(y,x)로 바꿔서 내부 (x=좌우,y=전방) 가정에 맞춤

        # Pure Pursuit
        self.declare_parameter('wheelbase', 1.0)                # [m]
        self.declare_parameter('ld_min', 2.5)                   # [m]
        self.declare_parameter('ld_max', 8.0)                   # [m]
        self.declare_parameter('ld_k_v', 0.03)                  # [m/(cmd)]  Ld = ld_min + k * speed_cmd (clamp to [ld_min,ld_max])
        self.declare_parameter('steer_limit_deg', 30.0)         # [deg]
        self.declare_parameter('steer_cnt_limit', 2000)
        self.declare_parameter('steer_lp_alpha', 0.6)           # 0~1 (클수록 최신값 반영)

        # 속도 정책 (기본 0이라 정지. 주행하려면 ros2 param set으로 바꿔줘)
        self.declare_parameter('speed_nominal', 30)              # 0~200
        self.declare_parameter('speed_min', 0)
        self.declare_parameter('speed_max',50)
        self.declare_parameter('brake_on_sharp_turn_deg', 18.0) # 급커브 임계

        # (옵션) 드럼/콘 회피
        self.declare_parameter('use_cones', False)              # True면 회피 기능 사용
        self.declare_parameter('cone_topic', '/drums/drum_info_down')
        self.declare_parameter('avoid_x_limit', 1.0)            # |x|< 이면 경로상 장애물 (x=좌우)
        self.declare_parameter('avoid_y_stop', 1.5)             # y< 이면 정지 (y=전방)
        self.declare_parameter('avoid_y_warn', 6.0)             # y< 이면 감속+바이어스
        self.declare_parameter('avoid_bias_deg', 8.0)           # 장애물 반대방향 추가 조향
        self.declare_parameter('slowdown_speed', 60)            # 감속 속도
        self.declare_parameter('keep_bias_time', 0.6)           # [s] 바이어스 유지 시간

        # 타이머
        self.declare_parameter('control_dt', 0.05)              # [s]

        ### [PID 추가] PID 파라미터
        self.declare_parameter('Kp_lat', 0.6)   # lateral error
        self.declare_parameter('Ki_lat', 0.0)
        self.declare_parameter('Kd_lat', 0.05)

        self.declare_parameter('Kp_head', 0.8)  # heading error
        self.declare_parameter('Ki_head', 0.0)
        self.declare_parameter('Kd_head', 0.1)

        # ===== QoS =====
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        # ===== I/O =====
        self.path_pts = []   # [(x=좌우, y=전방), ...]
        self.cones = []      # (옵션) 장애물 회피용

        center_topic = self.get_parameter('center_path_topic').value
        self.sub_center = self.create_subscription(Path, center_topic, self.center_cb, qos)

        # (옵션) 회피용 콘 구독
        if bool(self.get_parameter('use_cones').value):
            cone_topic = self.get_parameter('cone_topic').value
            self.sub_cone = self.create_subscription(ConeInfoArray, cone_topic, self.cone_cb, qos)
        else:
            self.sub_cone = None

        self.erp_pub = self.create_publisher(ErpCmdMsg, '/erp42_ctrl_cmd', qos)

        # 상태
        self.last_steer_cnt = 0
        self.last_speed_cmd = 30
        self.last_bias_sign = 0
        self.bias_until = 0.0

        dt = float(self.get_parameter('control_dt').value)
        self.timer = self.create_timer(dt, self.on_timer)

        ### [PID 추가] PID 컨트롤러 초기화
        self.pid_lat = PID(
            self.get_parameter('Kp_lat').value,
            self.get_parameter('Ki_lat').value,
            self.get_parameter('Kd_lat').value,
            setpoint=0.0
        )
        self.pid_lat.output_limits = (-10.0, 10.0)   # deg 보정 제한

        self.pid_head = PID(
            self.get_parameter('Kp_head').value,
            self.get_parameter('Ki_head').value,
            self.get_parameter('Kd_head').value,
            setpoint=0.0
        )
        self.pid_head.output_limits = (-10.0, 10.0)

        self.get_logger().info(f"✅ CenterPathFollower started (sub: {center_topic})")

    # ===== 콜백 =====
    def center_cb(self, msg: Path):
        swap_xy = bool(self.get_parameter('center_swap_xy').value)
        pts = []
        for ps in msg.poses:
            x = float(ps.pose.position.x)
            y = float(ps.pose.position.y)
            # Path는 보통 (x=전방, y=좌우) → 내부는 (x=좌우, y=전방)
            pts.append((y, x) if swap_xy else (x, y))
        self.path_pts = pts

    def cone_cb(self, msg: ConeInfoArray):
        self.cones = msg.cones if msg is not None else []

    # ===== 제어 루프 =====
    def on_timer(self):
        steer_deg = 0.0
        speed_cmd = int(self.get_parameter('speed_nominal').value)

        if len(self.path_pts) < 2:
            self._publish_cmd(0.0, 0)
            self.get_logger().warn("No center_path points -> stop")
            return

        # --- 기존 파라미터 로드 ---
        L   = float(self.get_parameter('wheelbase').value)
        ld_min = float(self.get_parameter('ld_min').value)
        ld_max = float(self.get_parameter('ld_max').value)
        ld_k_v = float(self.get_parameter('ld_k_v').value)
        steer_limit_deg = float(self.get_parameter('steer_limit_deg').value)
        steer_cnt_limit = int(self.get_parameter('steer_cnt_limit').value)
        steer_lp_alpha  = float(self.get_parameter('steer_lp_alpha').value)

        vmin = int(self.get_parameter('speed_min').value)
        vmax = int(self.get_parameter('speed_max').value)
        sharp_deg = float(self.get_parameter('brake_on_sharp_turn_deg').value)

        # --- Lookahead ---
        ld = max(ld_min, min(ld_max, ld_min + ld_k_v * max(self.last_speed_cmd, 0)))

        # --- Pure Pursuit 타겟 선택 ---
        tgt = None
        best_y = 1e9
        for (x, y) in self.path_pts:
            if y >= ld and y < best_y:
                best_y = y
                tgt = (x, y)
        if tgt is None:
            tgt = max(self.path_pts, key=lambda p: p[1])

        x_t, y_t = tgt
        kappa = (2.0 * x_t) / (ld * ld) if y_t > 1e-3 else 0.0
        delta = math.atan2(L * kappa, 1.0)
        steer_deg = math.degrees(delta)

        # --- 급커브 감속 ---
        if abs(steer_deg) > sharp_deg:
            speed_cmd = min(speed_cmd, int(self.get_parameter('slowdown_speed').value))

        ### [PID 추가] lateral error & heading error 기반 보정
        if len(self.path_pts) >= 2:
            x0, y0 = self.path_pts[0]
            x1, y1 = self.path_pts[1]

            a = y0 - y1
            b = x1 - x0
            c = x0*y1 - x1*y0

            lat_error = (a*0 + b*0 + c) / math.sqrt(a**2 + b**2 + 1e-6)
            path_yaw = math.atan2((y1-y0), (x1-x0))
            head_error = path_yaw  # 차량 yaw=0 기준

            delta_pid_lat = self.pid_lat(lat_error)
            delta_pid_head = self.pid_head(head_error)
            steer_deg += (delta_pid_lat + delta_pid_head)

        # --- 조향 제한 + LPF ---
        steer_deg = max(-steer_limit_deg, min(steer_limit_deg, steer_deg))
        steer_cnt = int(steer_deg / steer_limit_deg * steer_cnt_limit)
        steer_cnt = max(-steer_cnt_limit, min(steer_cnt_limit, steer_cnt))
        steer_cnt = int(steer_lp_alpha * steer_cnt + (1.0 - steer_lp_alpha) * self.last_steer_cnt)

        # --- 속도 제한 + LPF ---
        speed_cmd = max(vmin, min(vmax, speed_cmd))
        speed_cmd = int(0.7 * speed_cmd + 0.3 * self.last_speed_cmd)

        # --- 퍼블리시 ---
        self.last_steer_cnt = steer_cnt
        self.last_speed_cmd = speed_cmd
        self._publish_cmd(steer_deg, speed_cmd)

        self.get_logger().info(
            f"[center_path] ld={ld:.2f} tgt=({x_t:.2f},{y_t:.2f}) steer={steer_deg:+.1f}deg cnt={steer_cnt} speed={speed_cmd}"
        )

    # ===== Helpers =====
    def _publish_cmd(self, steer_deg_for_log: float, speed_cmd: int):
        steer_limit_deg = float(self.get_parameter('steer_limit_deg').value)
        steer_cnt_limit = int(self.get_parameter('steer_cnt_limit').value)

        steer_cnt = int(steer_deg_for_log / steer_limit_deg * steer_cnt_limit)
        steer_cnt = max(-steer_cnt_limit, min(steer_cnt_limit, steer_cnt))

        msg = ErpCmdMsg()
        msg.steer = steer_cnt
        msg.speed = int(speed_cmd)
        msg.gear = 0   # Forward (기본 0=Neutral → 수정)
        msg.brake = 0
        msg.e_stop = False
        self.erp_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PathFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("shutdown")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

