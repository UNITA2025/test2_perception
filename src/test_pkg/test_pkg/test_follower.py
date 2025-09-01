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


class CenterPathFollower(Node):
    def __init__(self):
        super().__init__('center_path_follower')

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
        self.declare_parameter('speed_nominal', 0)              # 0~200
        self.declare_parameter('speed_min', 0)
        self.declare_parameter('speed_max', 0)
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

        # ===== QoS =====
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10
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
        self.last_speed_cmd = 0
        self.last_bias_sign = 0
        self.bias_until = 0.0

        dt = float(self.get_parameter('control_dt').value)
        self.timer = self.create_timer(dt, self.on_timer)

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

        # --- 파라미터 로드 ---
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

        use_cones = bool(self.get_parameter('use_cones').value)
        avoid_x_limit = float(self.get_parameter('avoid_x_limit').value)
        avoid_y_stop  = float(self.get_parameter('avoid_y_stop').value)
        avoid_y_warn  = float(self.get_parameter('avoid_y_warn').value)
        avoid_bias_deg= float(self.get_parameter('avoid_bias_deg').value)
        slowdown_speed= int(self.get_parameter('slowdown_speed').value)
        keep_bias_time= float(self.get_parameter('keep_bias_time').value)

        # --- Lookahead ---
        ld = max(ld_min, min(ld_max, ld_min + ld_k_v * max(self.last_speed_cmd, 0)))

        # --- Pure Pursuit 타겟 선택 (y>=ld 중 가장 가까운 y, 없으면 가장 먼 점) ---
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
            speed_cmd = min(speed_cmd, slowdown_speed)

        # --- (옵션) 장애물 회피 ---
        if use_cones and self.cones:
            t_now = self.get_clock().now().nanoseconds * 1e-9
            nearest = None
            min_y = 1e9
            for c in self.cones:
                # c.x=좌우, c.y=전방
                if abs(c.x) <= avoid_x_limit and c.y > 0.0:
                    if c.y < min_y:
                        min_y = c.y
                        nearest = c

            if nearest is not None:
                if nearest.y <= avoid_y_stop:
                    speed_cmd = 0
                    self.last_bias_sign = 0
                    self.bias_until = t_now
                    self.get_logger().info(f"STOP drum x={nearest.x:.2f}, y={nearest.y:.2f}")
                elif nearest.y <= avoid_y_warn:
                    speed_cmd = min(speed_cmd, slowdown_speed)
                    bias_sign = 1 if nearest.x < 0.0 else -1  # 드럼이 좌(x<0)->우(+), 우(x>0)->좌(-)
                    steer_deg += bias_sign * avoid_bias_deg
                    self.last_bias_sign = bias_sign
                    self.bias_until = t_now + keep_bias_time
                    self.get_logger().info(
                        f"AVOID drum x={nearest.x:.2f}, y={nearest.y:.2f} -> bias {bias_sign:+d}*{avoid_bias_deg:.1f}deg"
                    )
            else:
                t_now = self.get_clock().now().nanoseconds * 1e-9
                if t_now < self.bias_until and self.last_bias_sign != 0:
                    steer_deg += self.last_bias_sign * avoid_bias_deg

        # --- 조향 제한 + LPF ---
        steer_deg = max(-steer_limit_deg, min(steer_limit_deg, steer_deg))
        steer_cnt = int(steer_deg / steer_limit_deg * steer_cnt_limit)
        steer_cnt = max(-steer_cnt_limit, min(steer_cnt_limit, steer_cnt))
        steer_cnt = int(steer_lp_alpha * steer_cnt + (1.0 - steer_lp_alpha) * self.last_steer_cnt)

        # --- 속도 제한 + 간단 LPF ---
        speed_cmd = max(vmin, min(vmax, speed_cmd))
        speed_cmd = int(0.7 * speed_cmd + 0.3 * self.last_speed_cmd)

        # --- 퍼블리시 ---
        self.last_steer_cnt = steer_cnt
        self.last_speed_cmd = speed_cmd
        self._publish_cmd(steer_deg, speed_cmd)

        self.get_logger().info(
            f"[center_path] ld={ld:.2f} tgt=({x_t:.2f},{y_t:.2f}) steer={steer_deg:+.1f}deg cnt={steer_cnt} speed={speed_cmd}"
            + (f" cones={len(self.cones)}" if use_cones else "")
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
        msg.gear = 0
        msg.brake = 0
        msg.e_stop = False
        self.erp_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CenterPathFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("shutdown")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
