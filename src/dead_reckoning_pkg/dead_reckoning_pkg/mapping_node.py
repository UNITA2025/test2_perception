#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FusionBuffer (full version, with use_msg_stamp_for_tf)
- /line_points, /drum_points 입력을 차량기준(base_link)에서 ROI로 필터링
- 전역(odom)으로 변환 후 시간창 누적 & 다운샘플링
- /lane_map, /drum_map 퍼블리시 (frame_id=odom)

ROS2 Humble / Python3
"""

from collections import deque
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration

from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2

from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped

# ========= 기본 프레임 이름 =========
DEFAULT_ODOM_FRAME  = "odom"
DEFAULT_BASE_FRAME  = "base_link"
DEFAULT_LIDAR_FRAME = "velodyne"

# ========= 기본 토픽 이름 =========
INPUT_LINE_TOPIC = "/lane_points"
INPUT_DRUM_TOPIC = "/drum_points"
OUTPUT_LINE_TOPIC = "/lane_map"
OUTPUT_DRUM_TOPIC = "/drum_map"

# ========= 수학/TF 유틸 =========
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

def pc2_to_xyz(msg: PointCloud2) -> np.ndarray:
    pts = [[p[0], p[1], p[2]] for p in pc2.read_points(msg, field_names=("x","y","z"), skip_nans=True)]
    return np.asarray(pts, dtype=np.float32) if pts else np.zeros((0,3), dtype=np.float32)

def voxel_downsample(xyz: np.ndarray, voxel=0.15):
    if xyz.shape[0] == 0:
        return xyz
    keys = np.floor(xyz / voxel).astype(np.int32)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return xyz[np.sort(idx)]


class Mapping(Node):
    def __init__(self):
        super().__init__('mapping_node')

        # ---------- 파라미터 ----------
        # 프레임
        self.odom_frame  = self.declare_parameter('odom_frame',  DEFAULT_ODOM_FRAME).get_parameter_value().string_value
        self.base_frame  = self.declare_parameter('base_frame',  DEFAULT_BASE_FRAME).get_parameter_value().string_value
        self.lidar_frame = self.declare_parameter('lidar_frame', DEFAULT_LIDAR_FRAME).get_parameter_value().string_value

        # 토픽
        self.topic_line_in  = self.declare_parameter('topic_line_in',  INPUT_LINE_TOPIC).get_parameter_value().string_value
        self.topic_drum_in  = self.declare_parameter('topic_drum_in',  INPUT_DRUM_TOPIC).get_parameter_value().string_value
        self.topic_lane_out = self.declare_parameter('topic_lane_out', OUTPUT_LINE_TOPIC).get_parameter_value().string_value
        self.topic_drum_out = self.declare_parameter('topic_drum_out', OUTPUT_DRUM_TOPIC).get_parameter_value().string_value

        # 윈도/필터/다운샘플
        self.window_sec = self.declare_parameter('window_sec', 2.0).get_parameter_value().double_value
        self.voxel      = self.declare_parameter('voxel', 0.15).get_parameter_value().double_value

        # ROI (base_link 기준)
        self.x_front = self.declare_parameter('x_front', 30.0).get_parameter_value().double_value
        self.y_band  = self.declare_parameter('y_band', 3.0).get_parameter_value().double_value
        self.z_min   = self.declare_parameter('z_min', -10.0).get_parameter_value().double_value
        self.z_max   = self.declare_parameter('z_max',  1.0).get_parameter_value().double_value

        # TF 타임아웃/폴백
        self.tf_timeout = Duration(seconds=self.declare_parameter('tf_timeout_sec', 0.5).get_parameter_value().double_value)
        self.tf_latest_ok_gap = self.declare_parameter('tf_latest_ok_if_within_sec', 0.3).get_parameter_value().double_value
        self.suppress_warn_first = self.declare_parameter('suppress_warn_first_sec', 3.0).get_parameter_value().double_value
        self.skip_if_far = self.declare_parameter('skip_if_tf_far', True).get_parameter_value().bool_value

        # ★ NEW: 메시지 스탬프 기반으로 TF를 조회할지, 항상 now()로 조회할지
        # 파생 포인트(/lane_points, /drum_points)라면 False 권장 → 최신 TF 사용
        self.use_msg_stamp_for_tf = self.declare_parameter(
            'use_msg_stamp_for_tf', True
        ).get_parameter_value().bool_value

        # TF 버퍼/리스너
        self.tfbuf = Buffer(Duration(seconds=20.0))  # bag 재생 고려 넉넉히
        self.tflis = TransformListener(self.tfbuf, self)

        # 워밍업: 필수 체인(odom↔base, base↔lidar) 준비 대기
        self.ready = False
        self._warm_hits = 0
        self._warm_timer = self.create_timer(0.1, self._warmup)

        # 로그 스로틀
        self._warns = 0
        self._warn_limit = 10
        self.boot_ts = self.get_clock().now()

        # 버퍼 (odom 좌표에서 누적)
        self.line_buf: deque = deque()  # (t_now, Nx3)
        self.drum_buf: deque = deque()

        # IO
        self.sub_line = self.create_subscription(PointCloud2, self.topic_line_in, self.cb_line, 10)
        self.sub_drum = self.create_subscription(PointCloud2, self.topic_drum_in, self.cb_drum, 10)
        self.pub_lane = self.create_publisher(PointCloud2, self.topic_lane_out, 10)
        self.pub_drum = self.create_publisher(PointCloud2, self.topic_drum_out, 10)

        # 주기 퍼블리시
        self.timer = self.create_timer(0.05, self.tick)  # 20 Hz

        self.get_logger().info(
            "FusionBuffer ready "
            f"(odom={self.odom_frame}, base={self.base_frame}, lidar={self.lidar_frame}, "
            f"use_msg_stamp_for_tf={self.use_msg_stamp_for_tf})\n"
            f"in: {self.topic_line_in}, {self.topic_drum_in}  |  out: {self.topic_lane_out}, {self.topic_drum_out}"
        )

    # ------------- Warmup -------------
    def _warmup(self):
        try:
            ok1 = self.tfbuf.can_transform(self.odom_frame, self.base_frame, Time(), Duration(seconds=0.5))
            ok2 = self.tfbuf.can_transform(self.base_frame, self.lidar_frame, Time(), Duration(seconds=0.5))
            if ok1 and ok2:
                self._warm_hits += 1
                if self._warm_hits >= 3:
                    self.ready = True
                    self.get_logger().info("TF chain(odom↔base, base↔lidar) ready. Start buffering.")
                    self._warm_timer.cancel()
        except Exception:
            pass

    # ------------- Helpers -------------
    def _apply_tf(self, xyz: np.ndarray, T: TransformStamped) -> np.ndarray:
        if xyz.shape[0] == 0:
            return xyz
        M = tf_to_mat(T)
        xyz_h = np.hstack([xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)])
        return (M @ xyz_h.T).T[:, :3]

    def _throttled_warn(self, msg: str):
        if (self.get_clock().now() - self.boot_ts) < Duration(seconds=self.suppress_warn_first):
            self.get_logger().debug(msg)
            return
        self._warns += 1
        if self._warns <= self._warn_limit:
            self.get_logger().warn(msg)
        else:
            self.get_logger().debug(msg)

    def _safe_time_from_msg(self, stamp_msg) -> Time:
        try:
            return Time.from_msg(stamp_msg)
        except Exception:
            sec = getattr(stamp_msg, 'sec', 0)
            nsec = getattr(stamp_msg, 'nanosec', 0)
            return Time(seconds=sec, nanoseconds=nsec)

    # ★ 범용: src → dst 좌표 변환 (stamp 또는 now() 사용 선택)
    def transform_xyz(self, xyz: np.ndarray, stamp_msg, src_frame: str, dst_frame: str):
        if xyz.shape[0] == 0:
            return None

        # --- 시각 결정 ---
        if self.use_msg_stamp_for_tf:
            t = self._safe_time_from_msg(stamp_msg)
            if t.nanoseconds == 0:
                # 메시지가 stamp=0이면 latest로 대체
                self.get_logger().debug("stamp=0 → using latest TF")
                t = Time()
        else:
            # 파생 포인트 및 누적용은 최신 TF 사용(경고폭탄 방지)
            t = self.get_clock().now()

        timeout = self.tf_timeout

        try:
            if self.tfbuf.can_transform(dst_frame, src_frame, t, timeout=timeout):
                T = self.tfbuf.lookup_transform(dst_frame, src_frame, t, timeout)
                return self._apply_tf(xyz, T)
            else:
                raise RuntimeError("can_transform=False at chosen time")
        except Exception as e1:
            # 최신으로 폴백 검토
            try:
                T_latest = self.tfbuf.lookup_transform(dst_frame, src_frame, Time(), timeout)
                t_latest = Time.from_msg(T_latest.header.stamp)
                # 비교는 디버그용 (선택적)
                if self.use_msg_stamp_for_tf:
                    t_msg = self._safe_time_from_msg(stamp_msg)
                    dt = abs(t_msg.nanoseconds - t_latest.nanoseconds) / 1e9
                else:
                    t_now = self.get_clock().now()
                    dt = abs(t_now.nanoseconds - t_latest.nanoseconds) / 1e9
            except Exception as e_latest:
                self._throttled_warn(f"TF lookup failed at chosen time ({e1}); latest lookup also failed: {e_latest}")
                return None

            if dt <= self.tf_latest_ok_gap:
                self.get_logger().debug(f"TF@chosen_time miss, fallback latest Δt={dt*1e3:.1f}ms")
                return self._apply_tf(xyz, T_latest)

            msg = f"TF@chosen_time miss; latest Δt={dt:.3f}s (>{self.tf_latest_ok_gap}s)"
            if self.skip_if_far:
                self._throttled_warn(msg + " → skip frame")
                return None
            else:
                self._throttled_warn(msg + " → use latest anyway")
                return self._apply_tf(xyz, T_latest)

    # ★ ROI는 base_link 기준(차량 전방/좌우/높이)
    def roi_filter_base(self, xyz_base: np.ndarray) -> np.ndarray:
        if xyz_base.shape[0] == 0:
            return xyz_base
        x, y, z = xyz_base[:, 0], xyz_base[:, 1], xyz_base[:, 2]
        m = (x >= 0.0) & (x <= self.x_front) & (np.abs(y) <= self.y_band) & (z >= self.z_min) & (z <= self.z_max)
        return xyz_base[m]

    def xyz_to_pc2(self, xyz: np.ndarray, frame_id: str) -> PointCloud2:
        header = Header()
        header.frame_id = frame_id
        header.stamp = self.get_clock().now().to_msg()  # 누적 결과이므로 now로 타임스탬프
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
        merged = voxel_downsample(merged, self.voxel)
        if merged.shape[0] == 0:
            return
        pub.publish(self.xyz_to_pc2(merged, self.odom_frame))

    # ------------- 콜백 -------------
    def cb_line(self, msg: PointCloud2):
        if not self.ready:
            return
        xyz_src = pc2_to_xyz(msg)
        if xyz_src.shape[0] == 0:
            return

        src_frame = msg.header.frame_id if msg.header.frame_id else self.lidar_frame

        # 1) src → base_link
        xyz_base = self.transform_xyz(xyz_src, msg.header.stamp, src_frame, self.base_frame)
        if xyz_base is None or xyz_base.shape[0] == 0:
            return

        # 2) ROI in base_link
        xyz_base = self.roi_filter_base(xyz_base)
        if xyz_base.shape[0] == 0:
            return

        # 3) base_link → odom (누적은 전역 좌표에서)
        xyz_odom = self.transform_xyz(xyz_base, msg.header.stamp, self.base_frame, self.odom_frame)
        if xyz_odom is None or xyz_odom.shape[0] == 0:
            return

        self.push_buf(self.line_buf, xyz_odom)

    def cb_drum(self, msg: PointCloud2):
        if not self.ready:
            return
        xyz_src = pc2_to_xyz(msg)
        if xyz_src.shape[0] == 0:
            return

        src_frame = msg.header.frame_id if msg.header.frame_id else self.lidar_frame

        xyz_base = self.transform_xyz(xyz_src, msg.header.stamp, src_frame, self.base_frame)
        if xyz_base is None or xyz_base.shape[0] == 0:
            return

        xyz_base = self.roi_filter_base(xyz_base)
        if xyz_base.shape[0] == 0:
            return

        xyz_odom = self.transform_xyz(xyz_base, msg.header.stamp, self.base_frame, self.odom_frame)
        if xyz_odom is None or xyz_odom.shape[0] == 0:
            return

        self.push_buf(self.drum_buf, xyz_odom)

    # ------------- 주기 퍼블리시 -------------
    def tick(self):
        if not self.ready:
            return
        self.merge_and_publish(self.line_buf, self.pub_lane)
        self.merge_and_publish(self.drum_buf, self.pub_drum)


def main():
    rclpy.init()
    n = Mapping()
    rclpy.spin(n)
    n.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
