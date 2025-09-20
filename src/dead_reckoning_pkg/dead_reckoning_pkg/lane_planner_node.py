#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math, numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

# ========= 기본 프레임 이름 =========
ODOM_FRAME = "odom"

# ========= 기본 토픽 이름 =========
INPUT_LINE_TOPIC = "/lane_map"
INPUT_DRUM_TOPIC = "/drum_map"
OUTPUT_PATH_TOPIC = "/center_path"

# ========= 유틸 =========
def pc2_to_xyz(msg: PointCloud2):
    pts = []
    for p in pc2.read_points(msg, field_names=("x","y","z"), skip_nans=True):
        pts.append([p[0],p[1],p[2]])
    if not pts:
        return np.zeros((0,3), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)

def smooth(y, k=5):
    if len(y) == 0:
        return y
    k = max(1, k); k = k + 1 - (k % 2)  # odd
    pad = k // 2
    yp = np.pad(y, (pad, pad), mode='edge')
    w = np.ones(k) / k
    return np.convolve(yp, w, mode='valid')

class LanePlanner(Node):
    def __init__(self):
        super().__init__('lane_planner_node')
        # ----- 파라미터 -----
        self.dx      = self.declare_parameter('dx', 1.0).value
        self.x_front = self.declare_parameter('x_front', 150.0).value
        self.y_band  = self.declare_parameter('y_band', 4.0).value
        self.lane_w  = self.declare_parameter('lane_width', 3.3).value
        self.max_off = self.declare_parameter('max_offset', 0.8).value
        self.rep_R   = self.declare_parameter('repulse_radius', 4.0).value
        self.rep_K   = self.declare_parameter('repulse_gain', 0.8).value

        # ----- 최신 버퍼 -----
        self.lane_pts = np.zeros((0,3), dtype=np.float32)
        self.drum_pts = np.zeros((0,3), dtype=np.float32)
        self.last_lane_header = None

        # ----- 구독/퍼블리시 -----
        self.sub_lane = self.create_subscription(PointCloud2, INPUT_LINE_TOPIC, self.cb_lane, 10)
        self.sub_drum = self.create_subscription(PointCloud2, INPUT_DRUM_TOPIC, self.cb_drum, 10)
        self.pub_path = self.create_publisher(Path, OUTPUT_PATH_TOPIC, 10)

    def cb_lane(self, msg: PointCloud2):
        # 최신 lane 저장
        self.lane_pts = pc2_to_xyz(msg)
        self.last_lane_header = msg.header  # frame_id, stamp 그대로 사용

        # 최신 drum 버퍼와 함께 즉시 경로 계산/퍼블리시
        path = self.build_path(self.lane_pts, self.drum_pts, hdr=msg.header)
        if path is not None:
            self.pub_path.publish(path)

    def cb_drum(self, msg: PointCloud2):
        # 최신 drum 저장 (경로 퍼블리시는 lane 콜백에서만)
        self.drum_pts = pc2_to_xyz(msg)

    def build_path(self, lane_xyz, drum_xyz, hdr):
        # 1) ROI
        if lane_xyz.shape[0] == 0:
            return None
        m = (lane_xyz[:,0] >= 0.0) & (lane_xyz[:,0] <= self.x_front) & (np.abs(lane_xyz[:,1]) <= self.y_band)
        L = lane_xyz[m]
        if L.shape[0] < 5:
            return None

        # 2) 좌/우 분리
        left  = L[L[:,1] >  0.0]
        right = L[L[:,1] <  0.0]

        # 3) x-bin 대표 y(중앙값)
        xs = np.arange(0.0, self.x_front + 1e-6, self.dx)
        def rep_line(pts):
            if pts.shape[0] == 0:
                return np.full(xs.shape, np.nan)
            out = np.full(xs.shape, np.nan)
            half = 0.5 * self.dx
            for i, x0 in enumerate(xs):
                sel = (pts[:,0] > x0 - half) & (pts[:,0] <= x0 + half)
                if np.any(sel):
                    out[i] = np.median(pts[sel, 1])
            return out

        yl = rep_line(left)
        yr = rep_line(right)

        # 4) 센터라인 y
        half_w = 0.5 * self.lane_w
        yc = np.empty_like(xs); yc[:] = np.nan
        for i in range(len(xs)):
            if not np.isnan(yl[i]) and not np.isnan(yr[i]):
                yc[i] = 0.5 * (yl[i] + yr[i])
            elif not np.isnan(yl[i]):
                yc[i] = yl[i] - half_w
            elif not np.isnan(yr[i]):
                yc[i] = yr[i] + half_w

        good = ~np.isnan(yc)
        if np.count_nonzero(good) < 2:
            return None
        xs_g = xs[good]
        yc_g = smooth(yc[good], k=7)

        # 5) 드럼 회피 오프셋
        off = np.zeros_like(yc_g)
        if drum_xyz.shape[0] > 0:
            D = drum_xyz[(drum_xyz[:,0] >= 0.0) & (drum_xyz[:,0] <= self.x_front)]
            for i, (x, y) in enumerate(zip(xs_g, yc_g)):
                near = D[np.abs(D[:,0] - x) <= self.rep_R]
                if near.shape[0] == 0:
                    continue
                for px, py, _ in near:
                    d = math.hypot(px - x, py - y)
                    if d < 1e-3:
                        continue
                    if d <= self.rep_R:
                        sgn = -np.sign(py - y)  # 드럼이 오른쪽이면 왼쪽으로 밀기
                        off[i] += sgn * self.rep_K * (1.0 / (d + 1e-2) - 1.0 / self.rep_R)
            off = np.clip(off, -self.max_off, self.max_off)
            off = smooth(off, k=9)

        y_final = yc_g + off

        # 6) Path 구성 — 입력 헤더 그대로 사용(시간/프레임 일관성)
        path = Path()
        path.header.frame_id = hdr.frame_id  # 보통 "odom"
        path.header.stamp    = hdr.stamp

        poses = []
        for i in range(len(xs_g)):
            x = xs_g[i]; y = y_final[i]
            yaw = 0.0
            if i < len(xs_g) - 1:
                dx = xs_g[i+1] - x
                dy = y_final[i+1] - y
                yaw = math.atan2(dy, dx)

            ps = PoseStamped()
            ps.header.frame_id = hdr.frame_id
            ps.header.stamp    = hdr.stamp   # 포즈에도 동일 stamp 부여(권장)
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = 0.0
            cy = math.cos(yaw * 0.5); sy = math.sin(yaw * 0.5)
            ps.pose.orientation.z = sy
            ps.pose.orientation.w = cy
            poses.append(ps)

        path.poses = poses
        return path

def main():
    rclpy.init()
    n = LanePlanner()
    rclpy.spin(n)
    n.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
