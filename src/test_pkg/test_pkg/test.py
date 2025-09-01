#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from math import sqrt, hypot
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2

# ==========================
# 기본 파라미터
INTENSITY_MIN_DEFAULT = 80.0
INTENSITY_MAX_DEFAULT = 150.0

# 거리 bin & 복셀
USE_3D_RANGE_DEFAULT = False                    # bin 구분용 거리: False=XY, True=XYZ
RANGE_BINS_DEFAULT   = [5.0, 10.0, 20.0]        # [0,5), [5,10), [10,20), [20,∞)
VOXEL_SIZES_DEFAULT  = [0.10, 0.10, 0.12, 0.18] # 각 bin 복셀 크기

# ROI
ROI_ENABLE_DEFAULT = True
ROI_MODE_DEFAULT   = 'box'   # 'none'|'box'|'sector'
ROI_X_MIN_DEFAULT  =  0.5
ROI_X_MAX_DEFAULT  =  7.0
ROI_Y_MIN_DEFAULT  = -3.0
ROI_Y_MAX_DEFAULT  =  3.0
ROI_Z_MIN_DEFAULT  = -10.0
ROI_Z_MAX_DEFAULT  =  0.0
ROI_R_MIN_DEFAULT  =  0.0
ROI_R_MAX_DEFAULT  = 30.0
ROI_TH_MIN_DEFAULT = -30.0
ROI_TH_MAX_DEFAULT =  30.0

# 대표점 병합(반경 기반)
REP_ENABLE_DEFAULT = True
REP_RADIUS_DEFAULT = 0.25     # [m]
REP_USE_3D_DEFAULT = False    # False: XY, True: XYZ

# 라인/드럼 분류 임계
Z_MAX_LINE_DEFAULT = -0.3     # z <= 이면 line, z > 이면 drum

# range 필드 계산 기준
RANGE_FIELD_METRIC_DEFAULT = 'xy'  # 'xy' | 'xyz'

# Topic
INPUT_CLOUD_DEFAULT  = '/velodyne_points'
LINE_OUT_TOPIC_DEF   = '/line_points'
DRUM_OUT_TOPIC_DEF   = '/drum_points'
# ==========================


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
        self.declare_parameter('range_bins',    RANGE_BINS_DEFAULT)
        self.declare_parameter('voxel_sizes',   VOXEL_SIZES_DEFAULT)

        self.declare_parameter('roi_enable', ROI_ENABLE_DEFAULT)
        self.declare_parameter('roi_mode',   ROI_MODE_DEFAULT)
        self.declare_parameter('roi_x_min',  ROI_X_MIN_DEFAULT)
        self.declare_parameter('roi_x_max',  ROI_X_MAX_DEFAULT)
        self.declare_parameter('roi_y_min',  ROI_Y_MIN_DEFAULT)
        self.declare_parameter('roi_y_max',  ROI_Y_MAX_DEFAULT)
        self.declare_parameter('roi_z_min',  ROI_Z_MIN_DEFAULT)
        self.declare_parameter('roi_z_max',  ROI_Z_MAX_DEFAULT)
        self.declare_parameter('roi_r_min',  ROI_R_MIN_DEFAULT)
        self.declare_parameter('roi_r_max',  ROI_R_MAX_DEFAULT)
        self.declare_parameter('roi_th_min', ROI_TH_MIN_DEFAULT)
        self.declare_parameter('roi_th_max', ROI_TH_MAX_DEFAULT)

        self.declare_parameter('rep_enable',  REP_ENABLE_DEFAULT)
        self.declare_parameter('rep_radius',  REP_RADIUS_DEFAULT)
        self.declare_parameter('rep_use_3d',  REP_USE_3D_DEFAULT)

        self.declare_parameter('z_max_line', Z_MAX_LINE_DEFAULT)
        self.declare_parameter('range_field_metric', RANGE_FIELD_METRIC_DEFAULT)

        in_topic  = self.get_parameter('input_topic').value
        line_out  = self.get_parameter('line_topic').value
        drum_out  = self.get_parameter('drum_topic').value

        # ---- I/O ----
        self.sub = self.create_subscription(PointCloud2, in_topic, self.callback, 10)
        self.pub_line = self.create_publisher(PointCloud2, line_out, 10)
        self.pub_drum = self.create_publisher(PointCloud2, drum_out, 10)

        self.get_logger().info(f"✅ lane_drum_compact (in:{in_topic} → out:{line_out}, {drum_out})")

    def callback(self, msg: PointCloud2):
        # --- Load params ---
        i_min = float(self.get_parameter('intensity_min').value)
        i_max = float(self.get_parameter('intensity_max').value)

        use_3d_bins = bool(self.get_parameter('use_3d_range').value)
        bins  = list(self.get_parameter('range_bins').value)
        vsize = list(self.get_parameter('voxel_sizes').value)

        roi_enable = bool(self.get_parameter('roi_enable').value)
        roi_mode   = str(self.get_parameter('roi_mode').value)
        x_min = float(self.get_parameter('roi_x_min').value)
        x_max = float(self.get_parameter('roi_x_max').value)
        y_min = float(self.get_parameter('roi_y_min').value)
        y_max = float(self.get_parameter('roi_y_max').value)
        z_min = float(self.get_parameter('roi_z_min').value)
        z_max = float(self.get_parameter('roi_z_max').value)
        r_min = float(self.get_parameter('roi_r_min').value)
        r_max = float(self.get_parameter('roi_r_max').value)
        th_min= float(self.get_parameter('roi_th_min').value)
        th_max= float(self.get_parameter('roi_th_max').value)

        rep_enable = bool(self.get_parameter('rep_enable').value)
        rep_r      = float(self.get_parameter('rep_radius').value)
        rep_3d     = bool(self.get_parameter('rep_use_3d').value)

        z_max_line = float(self.get_parameter('z_max_line').value)
        range_metric = str(self.get_parameter('range_field_metric').value).lower()  # 'xy'|'xyz'

        # --- Validate ---
        num_bins = len(bins) + 1
        if len(vsize) != num_bins:
            self.get_logger().error("voxel_sizes 길이는 len(range_bins)+1 이어야 합니다.")
            return

        # --- Read cloud ---
        pts = np.array([[p[0], p[1], p[2], p[3]] for p in pc2.read_points(
            msg, field_names=('x','y','z','intensity'), skip_nans=True
        )], dtype=float)
        if pts.size == 0:
            self._publish_empty_clouds(msg.header)
            return

        # --- Intensity filter ---
        pts = pts[(pts[:,3] >= i_min) & (pts[:,3] <= i_max)]
        if pts.size == 0:
            self._publish_empty_clouds(msg.header)
            return

        # --- ROI ---
        if roi_enable and roi_mode != 'none':
            pts = self._apply_roi(pts, roi_mode, x_min,x_max,y_min,y_max,z_min,z_max, r_min,r_max, th_min,th_max)
            if pts.size == 0:
                self._publish_empty_clouds(msg.header)
                return

        # --- Binning for voxel sizes ---
        r_xy  = np.hypot(pts[:,0], pts[:,1])
        r_xyz = np.sqrt(pts[:,0]**2 + pts[:,1]**2 + pts[:,2]**2)
        r_for_bins = r_xyz if use_3d_bins else r_xy

        bin_edges = np.array([0.0] + bins + [np.inf], dtype=float)
        idx = np.digitize(r_for_bins, bin_edges) - 1  # 0..num_bins-1

        # --- Voxel average per bin ---
        chunks = []
        for i in range(num_bins):
            mask = (idx == i)
            if not np.any(mask): continue
            vi = float(vsize[i])
            chunks.append(self._voxel_avg(pts[mask], vi))
        if not chunks:
            self._publish_empty_clouds(msg.header)
            return
        averaged = np.vstack(chunks)

        # --- Representative merge (optional) ---
        final_pts = averaged
        if rep_enable and final_pts.shape[0] > 0:
            final_pts = self._rep_merge(final_pts, rep_r, rep_3d)  # (N,4)

        # --- Split: line vs drum by Z ---
        line_mask = final_pts[:,2] <= z_max_line
        drum_mask = ~line_mask

        line_pts = final_pts[line_mask]
        drum_pts = final_pts[drum_mask]

        # --- Build range field for output (xy or xyz) ---
        def compute_range(arr):
            if arr.size == 0: return arr
            if range_metric == 'xyz':
                rr = np.sqrt(arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2)
            else:
                rr = np.hypot(arr[:,0], arr[:,1])
            # return x,y,z,range
            return np.column_stack([arr[:,0], arr[:,1], arr[:,2], rr])

        line_out = compute_range(line_pts)
        drum_out = compute_range(drum_pts)

        # --- Publish as compact PointCloud2 (x,y,z,range) ---
        fields = [
            PointField(name='x',     offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',     offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',     offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='range', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        self.pub_line.publish(pc2.create_cloud(msg.header, fields, line_out.tolist()))
        self.pub_drum.publish(pc2.create_cloud(msg.header, fields, drum_out.tolist()))

        self.get_logger().info(
            f"in:{pts.shape[0]} → avg:{averaged.shape[0]} → rep:{final_pts.shape[0]} | "
            f"line:{line_out.shape[0]} drum:{drum_out.shape[0]} (z_max_line={z_max_line:.2f}, range={range_metric})"
        )

    # ---------- helpers ----------
    def _apply_roi(self, pts, mode,
                   x_min,x_max,y_min,y_max,z_min,z_max,
                   r_min,r_max, th_min,th_max):
        if mode == 'box':
            m = (pts[:,0]>=x_min)&(pts[:,0]<=x_max)&(pts[:,1]>=y_min)&(pts[:,1]<=y_max)&(pts[:,2]>=z_min)&(pts[:,2]<=z_max)
            return pts[m]
        elif mode == 'sector':
            r = np.hypot(pts[:,0], pts[:,1])
            ang = np.degrees(np.arctan2(pts[:,1], pts[:,0]))
            if th_min <= th_max:
                aok = (ang >= th_min) & (ang <= th_max)
            else:
                aok = (ang >= th_min) | (ang <= th_max)
            m = (r>=r_min)&(r<=r_max)&aok&(pts[:,2]>=z_min)&(pts[:,2]<=z_max)
            return pts[m]
        else:
            return pts

    def _voxel_avg(self, pts, v):
        if pts.size == 0: return np.empty((0,4), dtype=float)
        coords = np.floor(pts[:,:3] / v).astype(np.int32)
        vox = {}
        for key, p in zip(map(tuple, coords), pts):
            vox.setdefault(key, []).append(p)
        out = []
        for group in vox.values():
            g = np.vstack(group)
            mean_xyz = g[:,:3].mean(axis=0)
            mean_i   = g[:,3].mean()
            out.append([mean_xyz[0], mean_xyz[1], mean_xyz[2], mean_i])
        return np.array(out, dtype=float) if out else np.empty((0,4), dtype=float)

    def _rep_merge(self, pts, radius, use_3d):
        if pts.shape[0] == 0: return pts
        inv = 1.0 / max(radius, 1e-6)

        def cell_of(x,y,z):
            return (int(np.floor(x*inv)), int(np.floor(y*inv))) if not use_3d else \
                   (int(np.floor(x*inv)), int(np.floor(y*inv)), int(np.floor(z*inv)))

        neigh = [(dx,dy) for dx in (-1,0,1) for dy in (-1,0,1)] if not use_3d else \
                [(dx,dy,dz) for dx in (-1,0,1) for dy in (-1,0,1) for dz in (-1,0,1)]

        sums=[]; counts=[]; cents=[]; cmap={}
        for x,y,z,i in pts:
            key = cell_of(x,y,z)
            cands=[]
            for off in neigh:
                k2 = tuple(a+b for a,b in zip(key, off))
                if k2 in cmap: cands.extend(cmap[k2])

            best=-1; bestd2=radius*radius
            for ci in cands:
                cx,cy,cz=cents[ci]
                d2 = (x-cx)**2 + (y-cy)**2 + ((z-cz)**2 if use_3d else 0.0)
                if d2<=bestd2:
                    best=ci; bestd2=d2

            if best>=0:
                sums[best][0]+=x; sums[best][1]+=y; sums[best][2]+=z; sums[best][3]+=i
                counts[best]+=1
                n=counts[best]
                cents[best][0]=sums[best][0]/n
                cents[best][1]=sums[best][1]/n
                cents[best][2]=sums[best][2]/n
                # 재배치
                for ck,lst in list(cmap.items()):
                    if best in lst:
                        lst.remove(best)
                        if not lst: del cmap[ck]
                new_key = cell_of(cents[best][0], cents[best][1], cents[best][2])
                cmap.setdefault(new_key, []).append(best)
            else:
                ci=len(sums)
                sums.append([x,y,z,i]); counts.append(1); cents.append([x,y,z])
                cmap.setdefault(key, []).append(ci)

        out=[]
        for (sx,sy,sz,si),n,(cx,cy,cz) in zip(sums,counts,cents):
            out.append([cx,cy,cz, si/n])
        return np.array(out, dtype=float)

    def _publish_empty_clouds(self, header):
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='range', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        empty = pc2.create_cloud(header, fields, [])
        self.pub_line.publish(empty)
        self.pub_drum.publish(empty)


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
