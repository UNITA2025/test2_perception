#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy

from sensor_msgs.msg import Image
from erp42_interfaces_pkg.msg import ErpCmdMsg

import cv2
from cv_bridge import CvBridge
import numpy as np

# -------------------------------
SUB_ROI_IMAGE   = '/roi_image'
PUB_ERP_CMD     = '/erp42_ctrl_cmd'
PUB_ROI_DEBUG   = '/roi_debug'       # 디버그용: 빨간 중앙선만
# -------------------------------

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

class DeadReckoningPlanner(Node):
    def __init__(self):
        super().__init__('dead_reckoning_planner')

        # ===== 파라미터 =====
        self.declare_parameter('steer_limit_deg', 30.0)
        self.declare_parameter('steer_limit_cnt', 2000)
        self.declare_parameter('max_speed_cmd',  80)      # 0~200
        self.declare_parameter('num_path_points', 20)     # 중앙선 샘플 개수
        self.declare_parameter('bin_thresh', 180)         # 이 값 이상이면 흰색(차선)
        self.declare_parameter('y_step_px',  max(1, 10))  # y 스캔 간격(px)
        self.declare_parameter('min_lane_width_px', 40)   # 좌/우로 잡힐 최소 간격
        self.declare_parameter('ema_alpha', 0.3)          # 중앙 x 보간용 EMA 계수

        self.ctrl_cmd_msg = ErpCmdMsg()
        self.bridge = CvBridge()

        # --- QoS 설정 ---
        img_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,   # 센서/이미지류 일반
            durability=QoSDurabilityPolicy.VOLATILE
        )
        debug_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=QoSReliabilityPolicy.RELIABLE,      # 뷰어가 RELIABLE 요구하는 경우 대응
            durability=QoSDurabilityPolicy.VOLATILE
        )
        cmd_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,      # 제어는 신뢰성 우선
            durability=QoSDurabilityPolicy.VOLATILE
        )

        # --- Subscribers ---
        self.roi_image_sub = self.create_subscription(
            Image, SUB_ROI_IMAGE, self.on_roi_image, img_qos
        )

        # --- Publishers ---
        self.erp_cmd_pub   = self.create_publisher(ErpCmdMsg, PUB_ERP_CMD, cmd_qos)
        self.debug_img_pub = self.create_publisher(Image, PUB_ROI_DEBUG, debug_qos)

        # 기본 상태
        self.last_speed_cmd = int(self.get_parameter('max_speed_cmd').value / 2)
        self.last_steer_cnt = 0

        # 중앙 x 추정값(EMA용)
        self.prev_center_x = None

        # 주기적으로 마지막 명령 재송신(20Hz)
        self.cmd_timer = self.create_timer(0.05, self.publish_last_cmd)

        self.get_logger().info('DeadReckoningPlanner up. ROI 디버그 퍼블리시(빨간 중앙선)!')

    # ---- 핵심: ROI에서 흰색 좌/우 차선 찾아 중앙선 좌표 뽑기 ----
    def extract_centerline(self, gray: np.ndarray):
        """
        gray: mono8 ROI (검정 배경, 흰색 차선)
        return: [(x_center, y), ...]  y는 아래->위 순서가 아니라 그리는 순서대로 반환
        """
        H, W = gray.shape[:2]
        N = int(self.get_parameter('num_path_points').value)
        y_step = int(self.get_parameter('y_step_px').value)
        bin_th = int(self.get_parameter('bin_thresh').value)
        min_gap = int(self.get_parameter('min_lane_width_px').value)
        alpha = float(self.get_parameter('ema_alpha').value)

        # 샘플할 y들 만들기 (아래쪽부터 위로)
        ys = list(range(H-1, -1, -y_step))
        if N > 0 and len(ys) > N:
            # 균등 샘플
            idxs = np.linspace(0, len(ys)-1, N).astype(int)
            ys = [ys[i] for i in idxs]

        centers = []
        for y in ys:
            row = gray[y, :]
            # 흰 픽셀 위치
            xs = np.flatnonzero(row >= bin_th)
            if xs.size < 2:
                # 흰 점이 1개 이하 → 중앙 못 구함
                cx = self.prev_center_x if self.prev_center_x is not None else (W//2)
            else:
                # 왼쪽 차선: 가장 왼쪽측 흰 픽셀 근방
                x_left = int(xs[0])
                # 오른쪽 차선: 가장 오른쪽측 흰 픽셀 근방
                x_right = int(xs[-1])
                if (x_right - x_left) < min_gap:
                    # 간격 너무 좁으면 신뢰 낮음 → 이전값 사용
                    cx = self.prev_center_x if self.prev_center_x is not None else (W//2)
                else:
                    cx_raw = 0.5 * (x_left + x_right)
                    # EMA 보간(튀는 프레임 억제)
                    if self.prev_center_x is None:
                        cx = cx_raw
                    else:
                        cx = alpha * cx_raw + (1.0 - alpha) * self.prev_center_x

            # 경계 클램프
            cx = int(clamp(int(round(cx)), 0, W-1))
            centers.append((cx, y))
            self.prev_center_x = cx  # 다음 y 행 계산에 누적(수직선에 가깝게 부드럽게)

        # 아래에서 위로 그리기 좋게 정순서로 뒤집기
        centers.reverse()
        return centers

    def on_roi_image(self, msg: Image):
        # ===== 1) ROI → OpenCV =====
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge 변환 에러: {e}')
            return

        h, w, _ = bgr.shape

        # gray & 이진 (흰 차선만 강조)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # ===== 2) 중앙선 계산: 좌/우 흰 차선 → 중앙 x =====
        center_pts = self.extract_centerline(gray)

        # ===== 3) 제어(로직은 유지: 더미로 직진) =====
        steering_angle_deg = 0.0
        speed_cmd = int(self.get_parameter('max_speed_cmd').value)
        steer_cnt = self.deg_to_count(steering_angle_deg)
        steer_cnt = clamp(steer_cnt, -self.get_parameter('steer_limit_cnt').value,
                          self.get_parameter('steer_limit_cnt').value)
        speed_cmd = clamp(speed_cmd, 0, 200)
        self.last_steer_cnt = int(steer_cnt)
        self.last_speed_cmd = int(speed_cmd)
        self.publish_cmd(self.last_steer_cnt, self.last_speed_cmd)

        # ===== 4) 디버그: 빨간 중앙선만 그려서 퍼블리시 =====
        dbg = bgr.copy()
        if len(center_pts) >= 2:
            pts = np.array(center_pts, dtype=np.int32)
            cv2.polylines(dbg, [pts], False, (0, 0, 255), 3, cv2.LINE_AA)  # 중앙선(빨강)
            for (x, y) in center_pts:
                cv2.circle(dbg, (x, y), 2, (0, 0, 255), -1, cv2.LINE_AA)

        try:
            dbg_msg = self.bridge.cv2_to_imgmsg(dbg, encoding='bgr8')
            dbg_msg.header.stamp = msg.header.stamp
            dbg_msg.header.frame_id = msg.header.frame_id
            self.debug_img_pub.publish(dbg_msg)
        except Exception as e:
            self.get_logger().error(f'debug 이미지 퍼블리시 실패: {e}')

    def deg_to_count(self, angle_deg: float) -> int:
        limit_deg = float(self.get_parameter('steer_limit_deg').value)
        limit_cnt = int(self.get_parameter('steer_limit_cnt').value)
        ratio = limit_cnt / limit_deg if limit_deg != 0 else 0.0
        return int(angle_deg * ratio)

    def publish_cmd(self, steer_cnt: int, speed_cmd: int):
        self.ctrl_cmd_msg.steer  = steer_cnt
        self.ctrl_cmd_msg.speed  = speed_cmd
        self.ctrl_cmd_msg.gear   = 0
        self.ctrl_cmd_msg.e_stop = False
        self.ctrl_cmd_msg.brake  = 0
        self.erp_cmd_pub.publish(self.ctrl_cmd_msg)

    def publish_last_cmd(self):
        self.publish_cmd(self.last_steer_cnt, self.last_speed_cmd)

def main(args=None):
    rclpy.init(args=args)
    node = DeadReckoningPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('shutdown!')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
