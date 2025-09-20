#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math, numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

ODOM_FRAME = "odom"

def pc2_to_xyz(msg: PointCloud2):
    pts = []
    for p in pc2.read_points(msg, field_names=("x","y","z"), skip_nans=True):
        pts.append([p[0],p[1],p[2]])
    if not pts:
        return np.zeros((0,3), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)

def smooth(y, k=5):
    if len(y)==0: return y
    k = max(1, k); k = k + 1 - (k%2)  # odd
    pad = k//2
    yp = np.pad(y, (pad,pad), mode='edge')
    w = np.ones(k)/k
    return np.convolve(yp, w, mode='valid')

class LanePathPlanner(Node):
    def __init__(self):
        super().__init__('lane_path_planner')
        # params
        self.dx      = self.declare_parameter('dx', 0.5).get_parameter_value().double_value
        self.x_front = self.declare_parameter('x_front', 45.0).get_parameter_value().double_value
        self.y_band  = self.declare_parameter('y_band', 6.0).get_parameter_value().double_value
        self.lane_w  = self.declare_parameter('lane_width', 3.3).get_parameter_value().double_value
        self.max_off = self.declare_parameter('max_offset', 0.8).get_parameter_value().double_value
        self.rep_R   = self.declare_parameter('repulse_radius', 2.0).get_parameter_value().double_value
        self.rep_K   = self.declare_parameter('repulse_gain', 0.8).get_parameter_value().double_value

        self.lane_pts = np.zeros((0,3), dtype=np.float32)
        self.drum_pts = np.zeros((0,3), dtype=np.float32)

        self.sub_lane = self.create_subscription(PointCloud2, '/lane_map', self.cb_lane, 10)
        self.sub_drum = self.create_subscription(PointCloud2, '/drum_map', self.cb_drum, 10)
        self.pub_path = self.create_publisher(Path, '/center_path', 10)

        self.timer = self.create_timer(0.1, self.tick)  # 10 Hz

    def cb_lane(self, msg): self.lane_pts = pc2_to_xyz(msg)
    def cb_drum(self, msg): self.drum_pts = pc2_to_xyz(msg)

    def tick(self):
        path = self.build_path(self.lane_pts, self.drum_pts)
        if path is not None:
            self.pub_path.publish(path)

    def build_path(self, lane_xyz, drum_xyz):
        # 1) ROI
        if lane_xyz.shape[0] == 0:
            return None
        m = (lane_xyz[:,0] >= 0.0) & (lane_xyz[:,0] <= self.x_front) & (np.abs(lane_xyz[:,1]) <= self.y_band)
        L = lane_xyz[m]
        if L.shape[0] < 20: ####
            return None

        # 2) 좌/우 분리 (y 부호, 히스테리시스 없이 단순화)
        left  = L[L[:,1] >  0.0]
        right = L[L[:,1] <  0.0]

        # 3) x-bin별 대표 y (중앙값)
        xs = np.arange(0.0, self.x_front + 1e-6, self.dx)
        def rep_line(pts):
            if pts.shape[0] == 0:
                return np.full(xs.shape, np.nan)
            out = np.full(xs.shape, np.nan)
            for i,x0 in enumerate(xs):
                sel = (pts[:,0] > x0-0.5*self.dx) & (pts[:,0] <= x0+0.5*self.dx)
                if np.any(sel):
                    out[i] = np.median(pts[sel,1])
            return out

        yl = rep_line(left)
        yr = rep_line(right)

        # 4) 센터 y: 양쪽 있으면 평균, 한쪽만 있으면 차선폭 보정
        half_w = 0.5*self.lane_w
        yc = np.empty_like(xs); yc[:] = np.nan
        for i in range(len(xs)):
            if not np.isnan(yl[i]) and not np.isnan(yr[i]):
                yc[i] = 0.5*(yl[i] + yr[i])
            elif not np.isnan(yl[i]):
                yc[i] = yl[i] - half_w
            elif not np.isnan(yr[i]):
                yc[i] = yr[i] + half_w

        # 유효 구간만 선택
        good = ~np.isnan(yc)
        if np.count_nonzero(good) < 5: ####
            return None
        xs_g = xs[good]; yc_g = smooth(yc[good], k=7)

        # 5) 드럼 회피 오프셋 (간단한 반발장)
        off = np.zeros_like(yc_g)
        if drum_xyz.shape[0] > 0:
            D = drum_xyz[(drum_xyz[:,0]>=0.0) & (drum_xyz[:,0]<=self.x_front)]
            for i,(x,y) in enumerate(zip(xs_g, yc_g)):
                near = D[np.abs(D[:,0]-x) <= self.rep_R]
                if near.shape[0]==0: continue
                for px,py,pz in near:
                    d = math.hypot(px - x, py - y)
                    if d < 1e-3: continue
                    if d <= self.rep_R:
                        sgn = -np.sign(py - y)  # 드럼이 오른쪽(py>y)이면 왼쪽(음수)으로 밀기
                        off[i] += sgn * self.rep_K * (1.0/(d+1e-2) - 1.0/self.rep_R)
            # 제한 & 스무딩
            off = np.clip(off, -self.max_off, self.max_off)
            off = smooth(off, k=9)

        y_final = yc_g + off

        # 6) Path 구성 (odom frame)
        path = Path()
        path.header.frame_id = ODOM_FRAME
        path.header.stamp = self.get_clock().now().to_msg()

        poses = []
        for i in range(len(xs_g)):
            x = xs_g[i]; y = y_final[i]
            yaw = 0.0
            if i < len(xs_g)-1:
                dx = xs_g[i+1]-x; dy = y_final[i+1]-y
                yaw = math.atan2(dy, dx)
            ps = PoseStamped()
            ps.header.frame_id = ODOM_FRAME
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = 0.0
            # yaw → quaternion (z축 회전)
            cy = math.cos(yaw*0.5); sy = math.sin(yaw*0.5)
            ps.pose.orientation.z = sy
            ps.pose.orientation.w = cy
            poses.append(ps)
        path.poses = poses
        return path

def main():
    rclpy.init()
    n = LanePathPlanner()
    rclpy.spin(n)
    n.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
