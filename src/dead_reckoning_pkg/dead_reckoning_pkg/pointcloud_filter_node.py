#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#==================================================#
# 기능(Function)
# - Velodyne LiDAR PointCloud2 전처리:
#   1) intensity 필터 → 2) ROI 필터 → 3) 거리(bin)별 voxel 평균화(xyz)
#   4) 2D 반경 기반 대표점 병합(xyz) → 5) z 기준 line/drum 분리
# - /lane_points, /drum_points 퍼블리시 (x,y,z,range(xy))
#
# 노드(Node): pointcloud_filter_node
#
# 수신(Subscribe)
# - /velodyne_points (sensor_msgs/PointCloud2)
#
# 송신(Publish)
# - /lane_points (sensor_msgs/PointCloud2) : z <= z_max_line
# - /drum_points (sensor_msgs/PointCloud2) : z >  z_max_line
#
# 파라미터(Parameters)
# - input_topic, line_topic, drum_topic
# - intensity_min/max
# - range_bins, voxel_sizes
# - roi_x/y/z_min/max
# - rep_enable, rep_radius
# - z_max_line
# - max_input_points
#
# TODO : 
# 최종 수정일: 2025.09.19
# 편집자: 이기현, 정선우
#==================================================#

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2

# ========= 기본 토픽 이름 =========
INPUT_CLOUD    = '/velodyne_points'
OUT_LINE_TOPIC = '/lane_points'
OUT_DRUM_TOPIC = '/drum_points'

# ---------- 기본 파라미터 ----------
INTENSITY_MIN_DEFAULT = 80.0
INTENSITY_MAX_DEFAULT = 150.0

RANGE_BINS_DEFAULT  = [2.0, 4.0, 6.0, 8.0]      # 0–2, 2–4, 4–6, 6–8, 8m+
VOXEL_SIZES_DEFAULT = [0.06, 0.08, 0.10, 0.12, 0.16]

ROI_X_MIN_DEFAULT, ROI_X_MAX_DEFAULT = 0.5, 8.0
ROI_Y_MIN_DEFAULT, ROI_Y_MAX_DEFAULT = -2.0, 3.0
ROI_Z_MIN_DEFAULT, ROI_Z_MAX_DEFAULT = -10.0, 0.0

REP_ENABLE_DEFAULT = True
REP_RADIUS_DEFAULT = 0.25

Z_MAX_LINE_DEFAULT = -0.5
MAX_INPUT_POINTS_DEFAULT = 200_000  # 프레임당 최대 처리 포인트(부하 제한)

def read_xyzi(msg: PointCloud2):
    """PointCloud2에서 (x,y,z,intensity) 읽기. intensity 없으면 None 반환."""
    names = [f.name for f in msg.fields]
    if 'intensity' not in names:
        return None
    arr = np.array(
        [ [p[0], p[1], p[2], float(p[3])] for p in pc2.read_points(
            msg, field_names=('x','y','z','intensity'), skip_nans=True)
        ],
        dtype=float
    )
    return arr

class PointcloudFilter(Node):
    def __init__(self):
        super().__init__('pointcloud_filter_node')

        # ---- Parameters ----
        self.declare_parameter('input_topic', INPUT_CLOUD)
        self.declare_parameter('line_topic',  OUT_LINE_TOPIC)
        self.declare_parameter('drum_topic',  OUT_DRUM_TOPIC)

        self.declare_parameter('intensity_min', INTENSITY_MIN_DEFAULT)
        self.declare_parameter('intensity_max', INTENSITY_MAX_DEFAULT)

        self.declare_parameter('range_bins',   RANGE_BINS_DEFAULT)
        self.declare_parameter('voxel_sizes',  VOXEL_SIZES_DEFAULT)

        self.declare_parameter('roi_x_min', ROI_X_MIN_DEFAULT)
        self.declare_parameter('roi_x_max', ROI_X_MAX_DEFAULT)
        self.declare_parameter('roi_y_min', ROI_Y_MIN_DEFAULT)
        self.declare_parameter('roi_y_max', ROI_Y_MAX_DEFAULT)
        self.declare_parameter('roi_z_min', ROI_Z_MIN_DEFAULT)
        self.declare_parameter('roi_z_max', ROI_Z_MAX_DEFAULT)

        self.declare_parameter('rep_enable',  REP_ENABLE_DEFAULT)
        self.declare_parameter('rep_radius',  REP_RADIUS_DEFAULT)

        self.declare_parameter('z_max_line', Z_MAX_LINE_DEFAULT)
        self.declare_parameter('max_input_points', MAX_INPUT_POINTS_DEFAULT)

        in_topic = self.get_parameter('input_topic').value
        line_out = self.get_parameter('line_topic').value
        drum_out = self.get_parameter('drum_topic').value

        self.sub = self.create_subscription(PointCloud2, in_topic, self.callback, 10)
        self.pub_line = self.create_publisher(PointCloud2, line_out, 10)
        self.pub_drum = self.create_publisher(PointCloud2, drum_out, 10)

        self.out_fields = [
            PointField(name='x',     offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',     offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',     offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='range', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        self.frame_count = 0
        self.get_logger().info(f"✅ pointcloud_filter_node (in:{in_topic} → out:{line_out}, {drum_out})")

    # ---------- helpers (xyz 전용) ----------
    def _voxel_avg(self, pts_xyz: np.ndarray, v: float) -> np.ndarray:
        """pts_xyz: (N,3) → voxel 평균 (N',3)"""
        if pts_xyz.size == 0:
            return np.empty((0,3), dtype=float)
        coords = np.floor(pts_xyz / v).astype(np.int32)
        vox = {}
        for key, p in zip(map(tuple, coords), pts_xyz):
            vox.setdefault(key, []).append(p)
        out = []
        for group in vox.values():
            g = np.vstack(group)
            out.append(g.mean(axis=0))
        return np.array(out, dtype=float) if out else np.empty((0,3), dtype=float)

    def _rep_merge(self, pts_xyz: np.ndarray, radius: float) -> np.ndarray:
        """2D(xy) 반경 기반 대표점 병합. 입력/출력: (N,3)"""
        if pts_xyz.shape[0] == 0:
            return pts_xyz
        inv = 1.0 / max(radius, 1e-6)
        def cell_of(x, y): return (int(np.floor(x*inv)), int(np.floor(y*inv)))
        neigh = [(dx,dy) for dx in (-1,0,1) for dy in (-1,0,1)]

        sums, counts, cents, cmap = [], [], [], {}
        for x, y, z in pts_xyz:
            key = cell_of(x, y)
            cands=[]
            for off in neigh:
                k2 = (key[0]+off[0], key[1]+off[1])
                if k2 in cmap: cands.extend(cmap[k2])
            best, bestd2 = -1, radius*radius
            for ci in cands:
                cx, cy, cz = cents[ci]
                d2 = (x-cx)**2 + (y-cy)**2
                if d2 <= bestd2:
                    best, bestd2 = ci, d2
            if best >= 0:
                sums[best][0]+=x; sums[best][1]+=y; sums[best][2]+=z
                counts[best]+=1; n=counts[best]
                cents[best][0]=sums[best][0]/n; cents[best][1]=sums[best][1]/n; cents[best][2]=sums[best][2]/n
                for ck,lst in list(cmap.items()):
                    if best in lst: lst.remove(best)
                    if not lst: del cmap[ck]
                cmap.setdefault(cell_of(cents[best][0], cents[best][1]), []).append(best)
            else:
                ci=len(sums); sums.append([x,y,z]); counts.append(1); cents.append([x,y,z]); cmap.setdefault(key,[]).append(ci)

        return np.array([c for c in cents], dtype=float)

    def _publish_empty_clouds(self, header):
        empty = pc2.create_cloud(header, self.out_fields, [])
        self.pub_line.publish(empty)
        self.pub_drum.publish(empty)

    # ---------- main callback ----------
    def callback(self, msg: PointCloud2):
        self.frame_count += 1

        # 파라미터
        i_min = float(self.get_parameter('intensity_min').value)
        i_max = float(self.get_parameter('intensity_max').value)
        bins  = list(self.get_parameter('range_bins').value)
        vsize = list(self.get_parameter('voxel_sizes').value)
        x_min = float(self.get_parameter('roi_x_min').value)
        x_max = float(self.get_parameter('roi_x_max').value)
        y_min = float(self.get_parameter('roi_y_min').value)
        y_max = float(self.get_parameter('roi_y_max').value)
        z_min = float(self.get_parameter('roi_z_min').value)
        z_max = float(self.get_parameter('roi_z_max').value)
        rep_enable = bool(self.get_parameter('rep_enable').value)
        rep_r      = float(self.get_parameter('rep_radius').value)
        z_max_line = float(self.get_parameter('z_max_line').value)
        max_in     = int(self.get_parameter('max_input_points').value)

        # bins/voxel 검증
        bins = list(np.asarray(bins, dtype=float))
        if len(bins) and (np.any(np.diff(bins) < 0) or bins[0] < 0):
            bins = sorted([b for b in bins if b >= 0])
            self.get_logger().warn("range_bins 정렬/음수 제거 후 적용")
        num_bins = len(bins) + 1
        if len(vsize) != num_bins or any(v <= 0 for v in vsize):
            self.get_logger().error("voxel_sizes 길이는 len(range_bins)+1 이고 모두 양수여야 합니다.")
            self._publish_empty_clouds(msg.header); return

        # 읽기 (x,y,z,intensity 필요)
        xyzi = read_xyzi(msg)
        if xyzi is None or xyzi.size == 0:
            self.get_logger().error("PointCloud2에 'intensity' 필드가 없습니다.")
            self._publish_empty_clouds(msg.header); return

        # 부하 제한
        if xyzi.shape[0] > max_in:
            xyzi = xyzi[np.random.choice(xyzi.shape[0], max_in, replace=False), :]

        # intensity 필터
        if i_min > i_max:
            self.get_logger().warn(f"intensity_min({i_min}) > intensity_max({i_max}) → swap")
            i_min, i_max = i_max, i_min
        mask_I = (xyzi[:,3] >= i_min) & (xyzi[:,3] <= i_max)
        if not np.any(mask_I):
            self._publish_empty_clouds(msg.header); return
        xyz = xyzi[mask_I, :3]  # 이후는 xyz만 사용

        # ROI
        m = ((xyz[:,0] >= x_min) & (xyz[:,0] <= x_max) &
             (xyz[:,1] >= y_min) & (xyz[:,1] <= y_max) &
             (xyz[:,2] >= z_min) & (xyz[:,2] <= z_max))
        xyz = xyz[m]
        if xyz.size == 0:
            self._publish_empty_clouds(msg.header); return

        # binning (항상 xy)
        r_xy = np.hypot(xyz[:,0], xyz[:,1])
        bin_edges = np.array([0.0] + bins + [np.inf], dtype=float)
        idx = np.digitize(r_xy, bin_edges) - 1  # 0..num_bins-1

        # voxel 평균화(xyz)
        chunks = []
        for i in range(num_bins):
            mi = (idx == i)
            if np.any(mi):
                chunks.append(self._voxel_avg(xyz[mi], float(vsize[i])))
        if not chunks:
            self._publish_empty_clouds(msg.header); return
        averaged = np.vstack(chunks)

        # 대표점 병합(2D)
        final_xyz = self._rep_merge(averaged, rep_r) if (rep_enable and averaged.shape[0] > 0) else averaged

        # z 분리
        line_xyz  = final_xyz[final_xyz[:,2] <= z_max_line]
        drum_xyz  = final_xyz[final_xyz[:,2] >  z_max_line]

        # range(xy) 추가
        def pack_xy_range(arr_xyz):
            if arr_xyz.size == 0:
                return arr_xyz
            rr = np.hypot(arr_xyz[:,0], arr_xyz[:,1])
            return np.column_stack([arr_xyz, rr])

        line_out = pack_xy_range(line_xyz)
        drum_out = pack_xy_range(drum_xyz)

        # 퍼블리시
        self.pub_line.publish(pc2.create_cloud(msg.header, self.out_fields, line_out.tolist()))
        self.pub_drum.publish(pc2.create_cloud(msg.header, self.out_fields, drum_out.tolist()))

        if self.frame_count % 10 == 0:
            self.get_logger().debug(
                f"in:{xyzi.shape[0]} → after_I+ROI:{xyz.shape[0]} → "
                f"avg:{averaged.shape[0]} → rep:{final_xyz.shape[0]} | "
                f"line:{line_out.shape[0]} drum:{drum_out.shape[0]} (z_max_line={z_max_line:.2f})"
            )

def main(args=None):
    rclpy.init(args=args)
    node = PointcloudFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()