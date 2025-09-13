#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

# ---------- 기본 파라미터 ----------
INTENSITY_MIN_DEFAULT = 70.0
INTENSITY_MAX_DEFAULT = 150.0

USE_3D_RANGE_DEFAULT  = False
RANGE_BINS_DEFAULT    = [3.6, 5.0, 8.0]   # → 총 구간은 len+1
VOXEL_SIZES_DEFAULT   = [0.10, 0.12, 0.15, 0.18]

ROI_X_MIN_DEFAULT, ROI_X_MAX_DEFAULT = 0.5, 6.0
ROI_Y_MIN_DEFAULT, ROI_Y_MAX_DEFAULT = -2.0, 3.0
ROI_Z_MIN_DEFAULT, ROI_Z_MAX_DEFAULT = -10.0, 0.0

REP_ENABLE_DEFAULT = True
REP_RADIUS_DEFAULT = 0.25
REP_USE_3D_DEFAULT = False

Z_MAX_LINE_DEFAULT = -0.5
RANGE_FIELD_METRIC_DEFAULT = 'xy'   # 'xy' or 'xyz'

INPUT_CLOUD_DEFAULT = '/velodyne_points'
LINE_OUT_TOPIC_DEF  = '/line_points'
DRUM_OUT_TOPIC_DEF  = '/drum_points'

MAX_INPUT_POINTS_DEFAULT = 200_000   # 프레임당 상한(부하 방지용)


def read_xyzI(msg: PointCloud2) -> np.ndarray:
    """
    PointCloud2에서 x,y,z와 intensity 유사 필드를 안전하게 읽는다.
    intensity/reflectivity/remission 중 하나가 있으면 사용,
    없으면 I=1.0으로 채워 반환.
    shape = (N,4) where columns = x,y,z,I
    """
    names = [f.name for f in msg.fields]
    for cand in ('intensity', 'reflectivity', 'remission'):
        if cand in names:
            arr = np.array([[p[0], p[1], p[2], float(p[3])]
                            for p in pc2.read_points(msg,
                                                     field_names=('x', 'y', 'z', cand),
                                                     skip_nans=True)],
                           dtype=float)
            return arr
    # fallback: intensity가 없을 때
    xyz = np.array([[p[0], p[1], p[2]]
                    for p in pc2.read_points(msg,
                                             field_names=('x', 'y', 'z'),
                                             skip_nans=True)],
                   dtype=float)
    if xyz.size == 0:
        return np.empty((0, 4), dtype=float)
    I = np.ones((xyz.shape[0], 1), dtype=float)
    return np.hstack([xyz, I])


class LaneDrumCompact(Node):
    def __init__(self):
        super().__init__('lane_drum_compact')

        # ---- Parameters ----
        self.declare_parameter('input_topic', INPUT_CLOUD_DEFAULT)
        self.declare_parameter('line_topic',  LINE_OUT_TOPIC_DEF)
        self.declare_parameter('drum_topic',  DRUM_OUT_TOPIC_DEF)

        self.declare_parameter('intensity_min', INTENSITY_MIN_DEFAULT)
        self.declare_parameter('intensity_max', INTENSITY_MAX_DEFAULT)

        self.declare_parameter('use_3d_range', USE_3D_RANGE_DEFAULT)
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
        self.declare_parameter('rep_use_3d',  REP_USE_3D_DEFAULT)

        self.declare_parameter('z_max_line', Z_MAX_LINE_DEFAULT)
        self.declare_parameter('range_field_metric', RANGE_FIELD_METRIC_DEFAULT)

        self.declare_parameter('max_input_points', MAX_INPUT_POINTS_DEFAULT)

        in_topic = self.get_parameter('input_topic').value
        line_out = self.get_parameter('line_topic').value
        drum_out = self.get_parameter('drum_topic').value

        # ---- QoS (필요 시 BEST_EFFORT로 드랍 감소) ----
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ---- I/O ----
        self.sub = self.create_subscription(PointCloud2, in_topic, self.callback, sensor_qos)
        self.pub_line = self.create_publisher(PointCloud2, line_out, 10)
        self.pub_drum = self.create_publisher(PointCloud2, drum_out, 10)

        # ---- Reusable fields (x,y,z,range) ----
        self.out_fields = [
            PointField(name='x',     offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',     offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',     offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='range', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        self.frame_count = 0
        self.get_logger().info(f"✅ lane_drum_compact (in:{in_topic} → out:{line_out}, {drum_out})")

    # ---------- helpers ----------
    def _voxel_avg(self, pts: np.ndarray, v: float) -> np.ndarray:
        """pts: (N,4) [x,y,z,I]  →  voxel 평균"""
        if pts.size == 0:
            return np.empty((0, 4), dtype=float)
        coords = np.floor(pts[:, :3] / v).astype(np.int32)
        vox = {}
        for key, p in zip(map(tuple, coords), pts):
            vox.setdefault(key, []).append(p)
        out = []
        for group in vox.values():
            g = np.vstack(group)
            mean_xyz = g[:, :3].mean(axis=0)
            mean_i   = g[:, 3].mean()
            out.append([mean_xyz[0], mean_xyz[1], mean_xyz[2], mean_i])
        return np.array(out, dtype=float) if out else np.empty((0, 4), dtype=float)

    def _rep_merge(self, pts: np.ndarray, radius: float, use_3d: bool) -> np.ndarray:
        """반경 기반 대표점 병합 (간단 그리드+근접). 입력/출력 (N,4)"""
        if pts.shape[0] == 0:
            return pts
        inv = 1.0 / max(radius, 1e-6)

        def cell_of(x, y, z):
            return (int(np.floor(x * inv)), int(np.floor(y * inv))) if not use_3d else \
                   (int(np.floor(x * inv)), int(np.floor(y * inv)), int(np.floor(z * inv)))

        neigh = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)] if not use_3d else \
                [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]

        sums, counts, cents, cmap = [], [], [], {}

        for x, y, z, i in pts:
            key = cell_of(x, y, z)
            cands = []
            for off in neigh:
                k2 = tuple(a + b for a, b in zip(key, off))
                if k2 in cmap:
                    cands.extend(cmap[k2])
            best = -1
            bestd2 = radius * radius
            for ci in cands:
                cx, cy, cz = cents[ci]
                d2 = (x - cx) ** 2 + (y - cy) ** 2 + ((z - cz) ** 2 if use_3d else 0.0)
                if d2 <= bestd2:
                    best = ci
                    bestd2 = d2
            if best >= 0:
                sums[best][0] += x; sums[best][1] += y; sums[best][2] += z; sums[best][3] += i
                counts[best] += 1
                n = counts[best]
                cents[best][0] = sums[best][0] / n
                cents[best][1] = sums[best][1] / n
                cents[best][2] = sums[best][2] / n
                # 재배치
                for ck, lst in list(cmap.items()):
                    if best in lst:
                        lst.remove(best)
                        if not lst:
                            del cmap[ck]
                new_key = cell_of(cents[best][0], cents[best][1], cents[best][2])
                cmap.setdefault(new_key, []).append(best)
            else:
                ci = len(sums)
                sums.append([x, y, z, i]); counts.append(1); cents.append([x, y, z])
                cmap.setdefault(key, []).append(ci)

        out = []
        for (sx, sy, sz, si), n, (cx, cy, cz) in zip(sums, counts, cents):
            out.append([cx, cy, cz, si / n])
        return np.array(out, dtype=float)

    def _publish_empty_clouds(self, header):
        empty = pc2.create_cloud(header, self.out_fields, [])
        self.pub_line.publish(empty)
        self.pub_drum.publish(empty)

    # ---------- main callback ----------
    def callback(self, msg: PointCloud2):
        self.frame_count += 1

        # --- Load params (프레임마다 읽어서 런타임 튜닝 가능) ---
        i_min = float(self.get_parameter('intensity_min').value)
        i_max = float(self.get_parameter('intensity_max').value)

        use_3d_bins = bool(self.get_parameter('use_3d_range').value)
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
        rep_3d     = bool(self.get_parameter('rep_use_3d').value)

        z_max_line   = float(self.get_parameter('z_max_line').value)
        range_metric = str(self.get_parameter('range_field_metric').value).lower()

        max_in = int(self.get_parameter('max_input_points').value)

        # --- Validate bins/voxel sizes ---
        bins = list(np.asarray(bins, dtype=float))
        if len(bins) and (np.any(np.diff(bins) < 0) or bins[0] < 0):
            bins = sorted([b for b in bins if b >= 0])
            self.get_logger().warn("range_bins 정렬/음수 제거 후 적용")

        num_bins = len(bins) + 1
        if len(vsize) != num_bins or any(v <= 0 for v in vsize):
            self.get_logger().error("voxel_sizes 길이는 len(range_bins)+1 이고 모두 양수여야 합니다.")
            self._publish_empty_clouds(msg.header)
            return

        # --- Read cloud (x,y,z,I) ---
        pts = read_xyzI(msg)
        if pts.size == 0:
            self._publish_empty_clouds(msg.header); return

        # 부하 제한
        if pts.shape[0] > max_in:
            sel = np.random.choice(pts.shape[0], max_in, replace=False)
            pts = pts[sel, :]

        # --- Intensity filter (실제 강도 있을 때만) ---
        has_real_I = not np.allclose(pts[:, 3], 1.0)
        if has_real_I:
            if i_min > i_max:
                self.get_logger().warn(f"intensity_min({i_min}) > intensity_max({i_max}) → swap")
                i_min, i_max = i_max, i_min
            pts = pts[(pts[:, 3] >= i_min) & (pts[:, 3] <= i_max)]
            if pts.size == 0:
                self._publish_empty_clouds(msg.header); return

        # --- ROI (box) ---
        pts = pts[(pts[:, 0] >= x_min) & (pts[:, 0] <= x_max) &
                  (pts[:, 1] >= y_min) & (pts[:, 1] <= y_max) &
                  (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)]
        if pts.size == 0:
            self._publish_empty_clouds(msg.header); return

        # --- Binning for voxel sizes ---
        r_xy  = np.hypot(pts[:, 0], pts[:, 1])
        r_xyz = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2 + pts[:, 2]**2)
        r_for_bins = r_xyz if use_3d_bins else r_xy

        bin_edges = np.array([0.0] + bins + [np.inf], dtype=float)
        idx = np.digitize(r_for_bins, bin_edges) - 1  # 0..num_bins-1

        # --- Voxel average per bin ---
        chunks = []
        for i in range(num_bins):
            mask = (idx == i)
            if not np.any(mask):
                continue
            vi = float(vsize[i])
            chunks.append(self._voxel_avg(pts[mask], vi))
        if not chunks:
            self._publish_empty_clouds(msg.header); return
        averaged = np.vstack(chunks)

        # --- Representative merge (optional) ---
        final_pts = self._rep_merge(averaged, rep_r, rep_3d) if (rep_enable and averaged.shape[0] > 0) else averaged

        # --- Split: line vs drum by Z ---
        line_mask = final_pts[:, 2] <= z_max_line
        line_pts  = final_pts[line_mask]
        drum_pts  = final_pts[~line_mask]

        # --- Build range field for output (xy or xyz) ---
        def to_xy_or_xyz(arr):
            if arr.size == 0:
                return arr
            rr = np.sqrt(arr[:, 0]**2 + arr[:, 1]**2 + (arr[:, 2]**2 if range_metric == 'xyz' else 0.0))
            return np.column_stack([arr[:, 0], arr[:, 1], arr[:, 2], rr])

        line_out = to_xy_or_xyz(line_pts)
        drum_out = to_xy_or_xyz(drum_pts)

        # --- Publish (입력 header 유지 → TF에는 영향 없음) ---
        self.pub_line.publish(pc2.create_cloud(msg.header, self.out_fields, line_out.tolist()))
        self.pub_drum.publish(pc2.create_cloud(msg.header, self.out_fields, drum_out.tolist()))

        if self.frame_count % 10 == 0:
            self.get_logger().debug(
                f"in:{pts.shape[0]} → avg:{averaged.shape[0]} → rep:{final_pts.shape[0]} | "
                f"line:{line_out.shape[0]} drum:{drum_out.shape[0]} "
                f"(z_max_line={z_max_line:.2f}, range={range_metric})"
            )

def main(args=None):
    rclpy.init(args=args)
    node = LaneDrumCompact()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
