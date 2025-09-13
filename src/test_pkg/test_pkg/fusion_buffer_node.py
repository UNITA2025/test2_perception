#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration

from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2

from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped

# ===== Frames =====
ODOM_FRAME  = "odom"
LIDAR_FRAME = "velodyne"


# ===== Math/TF utils =====
def quat_to_R(qx, qy, qz, qw):
    x, y, z, w = qx, qy, qz, qw
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),       1 - 2*(x*x + y*y)]
    ], dtype=np.float32)

def tf_to_mat(T: TransformStamped):
    t = T.transform.translation
    q = T.transform.rotation
    R = quat_to_R(q.x, q.y, q.z, q.w)
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = R
    M[:3,  3] = [t.x, t.y, t.z]
    return M

def pc2_to_xyz(msg: PointCloud2):
    pts = [[p[0], p[1], p[2]] for p in pc2.read_points(msg, field_names=("x","y","z"), skip_nans=True)]
    return np.asarray(pts, dtype=np.float32) if pts else np.zeros((0,3), dtype=np.float32)

def voxel_downsample(xyz: np.ndarray, voxel=0.15):
    if xyz.shape[0] == 0:
        return xyz
    keys = np.floor(xyz / voxel).astype(np.int32)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return xyz[np.sort(idx)]


# ===== Node =====
class FusionBuffer(Node):
    def __init__(self):
        super().__init__('fusion_buffer')

        # ----- Parameters -----
        self.window_sec = self.declare_parameter('window_sec', 8.0).get_parameter_value().double_value
        self.voxel      = self.declare_parameter('voxel', 0.15).get_parameter_value().double_value
        self.x_front    = self.declare_parameter('x_front', 10.0).get_parameter_value().double_value
        self.y_band     = self.declare_parameter('y_band', 6.0).get_parameter_value().double_value
        self.z_min      = self.declare_parameter('z_min', -10.0).get_parameter_value().double_value
        self.z_max      = self.declare_parameter('z_max',  1.0).get_parameter_value().double_value

        # ★ TF/로그 제어 파라미터 (추가)
        self.tf_timeout = Duration(
            seconds=self.declare_parameter('tf_timeout_sec', 0.5).get_parameter_value().double_value
        )
        self.tf_latest_ok_gap = self.declare_parameter(
            'tf_latest_ok_if_within_sec', 0.20 
        ).get_parameter_value().double_value
        self.suppress_warn_first = self.declare_parameter(
            'suppress_warn_first_sec', 3.0
        ).get_parameter_value().double_value
        self.skip_if_far = self.declare_parameter(
            'skip_if_tf_far', False
        ).get_parameter_value().bool_value

        # TF buffer/listener (캐시 넉넉히)
        self.tfbuf = Buffer(Duration(seconds=120.0))
        self.tflis = TransformListener(self.tfbuf, self)

        # 워밍업: TF 준비될 때까지 처리 보류
        self.ready = False
        self._warm_hits = 0
        self._warm_timer = self.create_timer(0.1, self._warmup)

        # 경고 스팸 억제
        self._warns = 0
        self._warn_limit = 10
        self.boot_ts = self.get_clock().now()

        # buffers
        self.line_buf = deque()  # (t_now, Nx3 in odom)
        self.drum_buf = deque()

        # IO
        self.sub_line = self.create_subscription(PointCloud2, '/line_points', self.cb_line, 10)
        self.sub_drum = self.create_subscription(PointCloud2, '/drum_points', self.cb_drum, 10)
        self.pub_lane = self.create_publisher(PointCloud2, '/lane_map', 10)
        self.pub_drum = self.create_publisher(PointCloud2, '/drum_map', 10)

        self.timer = self.create_timer(0.05, self.tick)  # 20 Hz

    # ---- warmup: TF 최신 변환이 들어오기 시작하면 시작 ----
    def _warmup(self):
        try:
            if self.tfbuf.can_transform(ODOM_FRAME, LIDAR_FRAME, Time(), Duration(seconds=0.5)):
                self._warm_hits += 1
                if self._warm_hits >= 3:  # 3번 연속 확인 후 시작
                    self.ready = True
                    self.get_logger().info("TF ready. Start buffering.")
                    self._warm_timer.cancel()
        except Exception:
            pass

    # ----- helpers -----
    def _apply_tf(self, xyz, T):
        if xyz.shape[0] == 0:
            return xyz
        M = tf_to_mat(T)
        xyz_h = np.hstack([xyz, np.ones((xyz.shape[0],1), dtype=np.float32)])
        return (M @ xyz_h.T).T[:, :3]

    def _throttled_warn(self, msg: str):
        # 부팅 초기 N초는 조용히
        if (self.get_clock().now() - self.boot_ts) < Duration(seconds=self.suppress_warn_first):
            self.get_logger().debug(msg)
            return
        self._warns += 1
        if self._warns <= self._warn_limit:
            self.get_logger().warn(msg)
        else:
            self.get_logger().debug(msg)

    def transform_xyz(self, xyz: np.ndarray, stamp_msg):
        if xyz.shape[0] == 0:
            return None

        # rclpy.Time로 변환 + stamp=0 방지
        try:
            t = Time.from_msg(stamp_msg)
        except Exception:
            t = Time(seconds=getattr(stamp_msg, 'sec', 0), nanoseconds=getattr(stamp_msg, 'nanosec', 0))

        # stamp가 0이면 최신으로 처리
        if t.nanoseconds == 0:
            self.get_logger().debug("stamp=0 → using latest TF")
            t = Time()

        timeout = self.tf_timeout

        # 1) 정확히 stamp 시각으로 시도
        try:
            if self.tfbuf.can_transform(ODOM_FRAME, LIDAR_FRAME, t, timeout=timeout):
                T = self.tfbuf.lookup_transform(ODOM_FRAME, LIDAR_FRAME, t, timeout)
                return self._apply_tf(xyz, T)
            else:
                raise RuntimeError("can_transform=False at stamp")
        except Exception as e1:
            # 최신 TF 가져와서 Δt 평가
            try:
                T_latest = self.tfbuf.lookup_transform(ODOM_FRAME, LIDAR_FRAME, Time(), timeout)
                t_latest = Time.from_msg(T_latest.header.stamp)
                dt = abs(t.nanoseconds - t_latest.nanoseconds) / 1e9
            except Exception as e_latest:
                self._throttled_warn(f"TF lookup failed at stamp ({e1}); latest lookup also failed: {e_latest}")
                return None

            # 2) Δt가 작으면 경고 없이 최신 TF로 폴백
            if dt <= self.tf_latest_ok_gap:
                self.get_logger().debug(f"TF@stamp miss, fallback latest Δt={dt*1e3:.1f}ms")
                return self._apply_tf(xyz, T_latest)

            # 3) 너무 멀면 (옵션) 스킵 or 최신 강행
            msg = f"TF@stamp miss; latest Δt={dt:.3f}s (>{self.tf_latest_ok_gap}s)"
            if self.skip_if_far:
                self._throttled_warn(msg + " → skip frame")
                return None
            else:
                self._throttled_warn(msg + " → use latest anyway")
                return self._apply_tf(xyz, T_latest)

    def roi_filter(self, xyz: np.ndarray):
        if xyz.shape[0] == 0:
            return xyz
        x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
        m = (x >= 0.0) & (x <= self.x_front) & (np.abs(y) <= self.y_band) & (z >= self.z_min) & (z <= self.z_max)
        return xyz[m]

    def xyz_to_pc2(self, xyz: np.ndarray, frame_id: str):
        header = Header()
        header.frame_id = frame_id
        header.stamp = self.get_clock().now().to_msg()
        fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        return pc2.create_cloud(header, fields, xyz.tolist())

    def push_buf(self, dq: deque, arr: np.ndarray):
        if arr is None or arr.shape[0] == 0:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        dq.append((now, arr))
        tcut = now - self.window_sec
        while dq and dq[0][0] < tcut:
            dq.popleft()

    def merge_and_publish(self, dq: deque, pub):
        if not dq:
            return
        merged = np.vstack([a for _, a in dq])
        merged = self.roi_filter(merged)
        merged = voxel_downsample(merged, self.voxel)
        if merged.shape[0] == 0:
            return
        pub.publish(self.xyz_to_pc2(merged, ODOM_FRAME))

    # ----- Callbacks -----
    def cb_line(self, msg: PointCloud2):
        if not self.ready:
            return
        xyz = pc2_to_xyz(msg)
        xyz_o = self.transform_xyz(xyz, msg.header.stamp)
        self.push_buf(self.line_buf, xyz_o)

    def cb_drum(self, msg: PointCloud2):
        if not self.ready:
            return
        xyz = pc2_to_xyz(msg)
        xyz_o = self.transform_xyz(xyz, msg.header.stamp)
        self.push_buf(self.drum_buf, xyz_o)

    def tick(self):
        if not self.ready:
            return
        self.merge_and_publish(self.line_buf, self.pub_lane)
        self.merge_and_publish(self.drum_buf, self.pub_drum)


def main():
    rclpy.init()
    n = FusionBuffer()
    rclpy.spin(n)
    n.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
