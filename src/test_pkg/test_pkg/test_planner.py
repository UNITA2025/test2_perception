#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
import numpy as np

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

# 필요하면 꺼도 됨
try:
    from interfaces_pkg.msg import PathPlanningResult
    HAS_PLANNER_MSG = True
except Exception:
    HAS_PLANNER_MSG = False


class CenterlinePlanner(Node):
    def __init__(self):
        super().__init__('centerline_planner')

        # ---------- Topics ----------
        self.declare_parameter('line_topic', '/line_points')
        self.declare_parameter('drum_topic', '/drum_points')
        self.declare_parameter('center_path_topic', '/center_path')
        self.declare_parameter('planner_topic', '/path_planning_result')  # PathPlanningResult (옵션)

        # ---------- Geometry / Filters ----------
        self.declare_parameter('y_left_is_neg', True)     # y<0 = 왼쪽
        self.declare_parameter('z_filter_enable', True)
        self.declare_parameter('z_max_for_path', -0.3)    # 이 값 이하만 라인 후보

        # x-bin 범위와 간격
        self.declare_parameter('x_min', 1.0)
        self.declare_parameter('x_max', 20.0)
        self.declare_parameter('bin_dx', 1.5)             # bin 폭

        # 라인 반폭(한쪽만 있을 때 center 추정)
        self.declare_parameter('lane_half_width', 1.5)    # [m]

        # 최소 샘플 수
        self.declare_parameter('min_points_per_side', 1)  # 좌/우 각 bin 최소 포인트
        self.declare_parameter('min_center_points', 4)    # 전체 center 최소 포인트

        # ---------- Smoothing ----------
        self.declare_parameter('use_polyfit', True)
        self.declare_parameter('poly_order', 2)           # 2가 무난
        self.declare_parameter('sample_step', 0.4)        # 피팅 후 샘플 간격

        # ---------- Drum avoidance (center 오프셋) ----------
        self.declare_parameter('drum_x_limit', 1.0)       # 경로 좌우 폭 제한(|y|<limit 이면 경로상)
        self.declare_parameter('drum_y_warn', 6.0)        # 이 y 이내면 회피 오프셋 적용
        self.declare_parameter('drum_bias_m', 0.5)        # [m] 중앙을 옆으로 민다 (드럼 반대 방향)

        # ---------- Planner msg (옵션) ----------
        self.declare_parameter('publish_planner_msg', True if HAS_PLANNER_MSG else False)
        # follower가 (x=좌우, y=전방) 기대 → 아래 True면 x_points=ys, y_points=xs 로 스왑해서 퍼블리시
        self.declare_parameter('planner_swap_xy', True)

        # ---------- IO ----------
        self._line_pts = []   # [(x,y,z), ...]
        self._line_hdr = None
        self._drum_pts = []
        self._drum_hdr = None

        line_topic   = self.get_parameter('line_topic').value
        drum_topic   = self.get_parameter('drum_topic').value
        center_topic = self.get_parameter('center_path_topic').value
        planner_topic= self.get_parameter('planner_topic').value

        self.sub_line = self.create_subscription(PointCloud2, line_topic, self._line_cb, 10)
        self.sub_drum = self.create_subscription(PointCloud2, drum_topic, self._drum_cb, 10)
        self.pub_path = self.create_publisher(Path, center_topic, 10)
        self.pub_planner = None
        if HAS_PLANNER_MSG:
            self.pub_planner = self.create_publisher(PathPlanningResult, planner_topic, 10)

        # 타이머로 주기적 생성 (두 토픽이 비동기로 들어와도 안정적)
        self.declare_parameter('planner_dt', 0.05)
        dt = float(self.get_parameter('planner_dt').value)
        self.timer = self.create_timer(dt, self._on_timer)

        self.get_logger().info(
            f"✅ CenterlinePlanner started | line:{line_topic} drum:{drum_topic} → path:{center_topic}"
            + (f" & planner:{planner_topic}" if self.pub_planner else "")
        )

    # ---------- Callbacks ----------
    def _line_cb(self, msg: PointCloud2):
        pts = []
        for x, y, z in pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True):
            pts.append((float(x), float(y), float(z)))
        self._line_pts = pts
        self._line_hdr = msg.header

    def _drum_cb(self, msg: PointCloud2):
        pts = []
        for x, y, z in pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True):
            pts.append((float(x), float(y), float(z)))
        self._drum_pts = pts
        self._drum_hdr = msg.header

    # ---------- Main loop ----------
    def _on_timer(self):
        if self._line_hdr is None:
            return

        # ----- load params -----
        y_left_is_neg = bool(self.get_parameter('y_left_is_neg').value)
        z_filter_enable = bool(self.get_parameter('z_filter_enable').value)
        z_max = float(self.get_parameter('z_max_for_path').value)

        x_min = float(self.get_parameter('x_min').value)
        x_max = float(self.get_parameter('x_max').value)
        bin_dx = float(self.get_parameter('bin_dx').value)

        lane_half = float(self.get_parameter('lane_half_width').value)
        min_side = int(self.get_parameter('min_points_per_side').value)
        min_center = int(self.get_parameter('min_center_points').value)

        use_polyfit = bool(self.get_parameter('use_polyfit').value)
        poly_order = int(self.get_parameter('poly_order').value)
        sample_step = float(self.get_parameter('sample_step').value)

        drum_x_limit = float(self.get_parameter('drum_x_limit').value)
        drum_y_warn  = float(self.get_parameter('drum_y_warn').value)
        drum_bias_m  = float(self.get_parameter('drum_bias_m').value)

        publish_planner = bool(self.get_parameter('publish_planner_msg').value)
        planner_swap_xy = bool(self.get_parameter('planner_swap_xy').value)

        # ----- prepare points -----
        pts = np.array(self._line_pts, dtype=float) if self._line_pts else np.empty((0, 3))

        self.get_logger().info(f"[dbg] line pts in: {pts.shape[0]}")

        if pts.shape[0] == 0:
            self._publish_empty_path(self._line_hdr)
            return

        if z_filter_enable:
            pts = pts[pts[:, 2] <= z_max]
            self.get_logger().info(f"[dbg] after z<= {z_max:.2f}: {pts.shape[0]}")
            if pts.shape[0] == 0:
                self._publish_empty_path(self._line_hdr)
                return

        # 좌/우 분리 (y 부호)
        if y_left_is_neg:
            left = pts[pts[:, 1] <  0.0]
            right= pts[pts[:, 1] >  0.0]
        else:
            left = pts[pts[:, 1] >  0.0]
            right= pts[pts[:, 1] <  0.0]

        self.get_logger().info(f"[dbg] L:{left.shape[0]}  R:{right.shape[0]}")

        # ----- x-binning -----
        xs = np.arange(x_min, x_max + 1e-6, bin_dx)
        centers = []  # (x, y_center, z_med)

        for xb in xs:
            x0, x1 = xb, xb + bin_dx
            # 각 bin에 속하는 좌/우 포인트
            L = left[(left[:, 0] >= x0) & (left[:, 0] < x1)]
            R = right[(right[:, 0] >= x0) & (right[:, 0] < x1)]

            have_L = L.shape[0] >= min_side
            have_R = R.shape[0] >= min_side

            if not have_L and not have_R:
                continue

            # 각 측 median y, z
            yLc = np.median(L[:, 1]) if have_L else None
            zLc = np.median(L[:, 2]) if have_L else None
            yRc = np.median(R[:, 1]) if have_R else None
            zRc = np.median(R[:, 2]) if have_R else None

            if have_L and have_R:
                yC = 0.5 * (yLc + yRc)
                zC = 0.5 * (zLc + zRc)
            elif have_L:
                # 왼쪽만: 차선 반폭으로 보정
                yC = yLc + (lane_half if y_left_is_neg else -lane_half)
                zC = zLc
            else:
                # 오른쪽만: 차선 반폭으로 보정
                yC = yRc - (lane_half if y_left_is_neg else -lane_half)
                zC = zRc

            centers.append((0.5 * (x0 + x1), float(yC), float(zC)))

        self.get_logger().info(f"[dbg] x∈[{x_min},{x_max}] dx={bin_dx} → centers_try={len(centers)} (min={min_center})")

        if len(centers) < min_center:
            self._publish_empty_path(self._line_hdr)
            return

        centers = np.array(centers, dtype=float)  # Nx3
        xs_c = centers[:, 0]
        ys_c = centers[:, 1]
        z_med = float(np.median(centers[:, 2]))

        # ----- drum avoidance (반발 오프셋) -----
        if len(self._drum_pts) > 0:
            drums = np.array(self._drum_pts, dtype=float)  # Mx3
            # 드럼 중 전방( x in [x_min, x_max] ) & |y|<limit & x>0
            sel = (drums[:, 0] >= x_min) & (drums[:, 0] <= x_max) & (np.abs(drums[:, 1]) <= drum_x_limit)
            drums = drums[sel]
            if drums.shape[0] > 0:
                # 각 center x에 대해 가까운 드럼의 y, x 거리보고 오프셋
                for i in range(xs_c.shape[0]):
                    xq = xs_c[i]
                    # 드럼들 중 앞쪽( x>=xq ) & 경고 거리 이내
                    cand = drums[(drums[:, 0] >= xq) & ((drums[:, 0] - xq) <= drum_y_warn)]
                    if cand.shape[0] == 0:
                        continue
                    # 가장 가까운 드럼 선택
                    j = np.argmin(cand[:, 0] - xq)
                    dy = cand[j, 1]  # +: 좌, -: 우 (ROS 일반 좌표계: x=전방, y=좌측)
                    # 드럼이 왼쪽(y>0)이면 중앙을 약간 오른쪽(음수)으로, 반대도 동일
                    bias = -np.sign(dy) * drum_bias_m
                    ys_c[i] += bias

        # ----- smoothing (polyfit) -----
        if use_polyfit and xs_c.shape[0] >= (poly_order + 1) and (xs_c.max() - xs_c.min()) > sample_step * 0.5:
            try:
                coeffs = np.polyfit(xs_c, ys_c, poly_order)
                poly = np.poly1d(coeffs)
                xs_samp = np.arange(xs_c.min(), xs_c.max() + sample_step * 0.5, sample_step)
                ys_samp = poly(xs_samp)
                xs_out, ys_out = xs_samp, ys_samp
            except Exception:
                xs_out, ys_out = xs_c, ys_c
        else:
            xs_out, ys_out = xs_c, ys_c

        # ----- publish Path -----
        path = Path()
        path.header = self._line_hdr
        for x, y in zip(xs_out, ys_out):
            ps = PoseStamped()
            ps.header = self._line_hdr
            ps.pose.position.x = float(x)   # x=전방
            ps.pose.position.y = float(y)   # y=좌/우
            ps.pose.position.z = z_med      # 고정/중앙 z
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.pub_path.publish(path)

        # ----- (옵션) PathPlanningResult -----
        if publish_planner and self.pub_planner is not None:
            if HAS_PLANNER_MSG and len(xs_out) >= min_center:
                msg = PathPlanningResult()
                if planner_swap_xy:
                    # follower가 (x=좌우, y=전방) 기대 → x_points=ys, y_points=xs
                    msg.x_points = [float(y) for y in ys_out]
                    msg.y_points = [float(x) for x in xs_out]
                else:
                    msg.x_points = [float(x) for x in xs_out]
                    msg.y_points = [float(y) for y in ys_out]
                self.pub_planner.publish(msg)

        self.get_logger().info(
            f"center N={len(xs_out)} | drums_used={len(self._drum_pts)} | z_max={z_max:.2f} | lane_half={lane_half:.2f}"
        )

    # ---------- helpers ----------
    def _publish_empty_path(self, header):
        path = Path()
        path.header = header
        self.pub_path.publish(path)
        # planner msg는 빈 경로 미발행(원한다면 여기서도 비우거나 이전 유지 정책 택일)


def main(args=None):
    rclpy.init(args=args)
    node = CenterlinePlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("shutdown")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
