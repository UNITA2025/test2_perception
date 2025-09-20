#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
import numpy as np

from sensor_msgs_py import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import ColorRGBA
from nav_msgs.msg import Path


class LinePointsMarkers(Node):
    def __init__(self):
        super().__init__('line_points_markers')

        # -------- Parameters --------
        self.declare_parameter('line_topic', '/line_points')
        self.declare_parameter('drum_topic', '/drum_points')
        self.declare_parameter('output_topic', '/line_points_markers')

        # 좌/우 분리 관련
        self.declare_parameter('y_left_is_neg', False)   # True면 y<0이 좌측
        self.declare_parameter('y_split_offset', 0.0)    # 분리 기준 y 오프셋
        self.declare_parameter('x_forward_min', 0.0)     # 앞쪽만 쓰려면 >0로

        # 라인(좌/우) 점 마커 설정
        self.declare_parameter('line_point_scale', 0.10)          # sphere용 (계속 사용)
        self.declare_parameter('line_left_color',  [0.2, 0.6, 1.0, 1.0])  # 파랑
        self.declare_parameter('line_right_color', [0.2, 1.0, 0.4, 1.0])  # 초록

        # 라인 스트립(연결선) 스타일
        self.declare_parameter('strip_width', 0.06)               # LINE_STRIP 두께[m]
        self.declare_parameter('strip_alpha', 1.0)                # 불투명도(0~1)

        # 센터라인 (옵션)
        self.declare_parameter('draw_centerline', False)
        self.declare_parameter('center_color', [1.0, 1.0, 0.2, 1.0])       # 노랑
        self.declare_parameter('center_strip_width', 0.08)
        self.declare_parameter('center_path_topic', '')                     # 비우면 Path 퍼블리시 안함

        # 드럼 점 마커 설정
        self.declare_parameter('drum_point_scale', 0.12)
        self.declare_parameter('drum_point_color', [1.0, 0.2, 0.2, 1.0])  # 빨강

        # 과밀 시 샘플링
        self.declare_parameter('line_decimate', 1)
        self.declare_parameter('drum_decimate', 1)

        # 라인 정렬/스무딩/리샘플 파라미터
        self.declare_parameter('min_points_for_strip', 6)   # 이보다 적으면 스트립 그리지 않음
        self.declare_parameter('sort_by', 'x')              # 'x' 또는 'range' (전방 정렬기준)
        self.declare_parameter('smooth_window', 5)          # 이동평균 윈도(홀수 권장, 1이면 미적용)
        self.declare_parameter('resample_ds', 0.10)         # 선따기 간격[m], <=0 이면 리샘플 안함

        line_topic = self.get_parameter('line_topic').value
        drum_topic = self.get_parameter('drum_topic').value
        out_topic  = self.get_parameter('output_topic').value

        self._line_pts = []   # [(x,y,z,range), ...]
        self._line_hdr = None
        self._drum_pts = []
        self._drum_hdr = None

        self.sub_line = self.create_subscription(PointCloud2, line_topic, self._line_cb, 10)
        self.sub_drum = self.create_subscription(PointCloud2, drum_topic, self._drum_cb, 10)
        self.pub_markers = self.create_publisher(MarkerArray, out_topic, 10)

        # (옵션) center Path 퍼블리셔
        center_path_topic = self.get_parameter('center_path_topic').value
        self.pub_center_path = None
        if isinstance(center_path_topic, str) and len(center_path_topic) > 0:
            self.pub_center_path = self.create_publisher(Path, center_path_topic, 10)

        self.get_logger().info(
            f"✅ line_points_markers started (line: {line_topic}, drum: {drum_topic} → markers: {out_topic}"
            + (f", center_path: {center_path_topic}" if self.pub_center_path else "") + ")"
        )

    # -------- Callbacks --------
    def _line_cb(self, msg: PointCloud2):
        decimate = max(1, int(self.get_parameter('line_decimate').value))
        pts = self._read_xyzr(msg)
        if decimate > 1:
            pts = pts[::decimate]
        self._line_pts = pts
        self._line_hdr = msg.header
        self._publish_all()

    def _drum_cb(self, msg: PointCloud2):
        decimate = max(1, int(self.get_parameter('drum_decimate').value))
        pts = self._read_xyzr(msg)
        if decimate > 1:
            pts = pts[::decimate]
        self._drum_pts = pts
        self._drum_hdr = msg.header
        self._publish_all()

    # -------- Main Publisher --------
    def _publish_all(self):
        arr = MarkerArray()
        ns = "line_drum_points"

        # ====== 분리 기준 파라미터 ======
        y_left_is_neg = bool(self.get_parameter('y_left_is_neg').value)
        y_off         = float(self.get_parameter('y_split_offset').value)
        x_forward_min = float(self.get_parameter('x_forward_min').value)

        # ====== 라인 좌/우 분리 ======
        line_left_pts, line_right_pts = [], []
        if self._line_pts:
            for x, y, z, r in self._line_pts:
                if x < x_forward_min:
                    continue
                y_adj = y - y_off
                is_left = (y_adj < 0.0) if y_left_is_neg else (y_adj > 0.0)
                (line_left_pts if is_left else line_right_pts).append((x, y, z))

        # ====== 점 마커 (기존처럼 표시) ======
        # Left points (id=0)
        arr.markers.append(self._make_sphere_list_marker(
            header=self._line_hdr, ns=ns, mid=0,
            points=line_left_pts,
            scale=float(self.get_parameter('line_point_scale').value),
            color=self._as_color(self.get_parameter('line_left_color').value)
        ))
        # Right points (id=1)
        arr.markers.append(self._make_sphere_list_marker(
            header=self._line_hdr, ns=ns, mid=1,
            points=line_right_pts,
            scale=float(self.get_parameter('line_point_scale').value),
            color=self._as_color(self.get_parameter('line_right_color').value)
        ))
        # Drum points (id=2)
        drum_pts_xyz = [(x,y,z) for (x,y,z,_) in self._drum_pts] if self._drum_pts else []
        arr.markers.append(self._make_sphere_list_marker(
            header=self._drum_hdr, ns=ns, mid=2,
            points=drum_pts_xyz,
            scale=float(self.get_parameter('drum_point_scale').value),
            color=self._as_color(self.get_parameter('drum_point_color').value)
        ))

        # ====== 라인 스트립(연결선) ======
        min_pts = int(self.get_parameter('min_points_for_strip').value)
        strip_w = float(self.get_parameter('strip_width').value)
        strip_alpha = float(self.get_parameter('strip_alpha').value)
        sort_by = str(self.get_parameter('sort_by').value).lower()
        smooth_window = int(self.get_parameter('smooth_window').value)
        ds = float(self.get_parameter('resample_ds').value)

        # Left strip (id=10)
        left_strip = self._build_strip(line_left_pts, min_pts, sort_by, smooth_window, ds)
        arr.markers.append(self._make_line_strip_marker(
            header=self._line_hdr, ns=ns, mid=10,
            points=left_strip,
            width=strip_w, color=self._with_alpha(self._as_color(self.get_parameter('line_left_color').value), strip_alpha)
        ))
        # Right strip (id=11)
        right_strip = self._build_strip(line_right_pts, min_pts, sort_by, smooth_window, ds)
        arr.markers.append(self._make_line_strip_marker(
            header=self._line_hdr, ns=ns, mid=11,
            points=right_strip,
            width=strip_w, color=self._with_alpha(self._as_color(self.get_parameter('line_right_color').value), strip_alpha)
        ))

        # ====== 센터라인 (옵션) ======
        draw_center = bool(self.get_parameter('draw_centerline').value)
        center_strip = []
        if draw_center and (len(left_strip) >= 2 and len(right_strip) >= 2):
            center_strip = self._build_centerline(left_strip, right_strip, ds if ds > 0 else 0.10)
        # Center strip (id=12)
        arr.markers.append(self._make_line_strip_marker(
            header=self._line_hdr, ns=ns, mid=12,
            points=center_strip,
            width=float(self.get_parameter('center_strip_width').value),
            color=self._as_color(self.get_parameter('center_color').value)
        ))

        # ====== 퍼블리시 ======
        if arr.markers:
            self.pub_markers.publish(arr)

        # (옵션) nav_msgs/Path 퍼블리시
        if self.pub_center_path and center_strip:
            path_msg = Path()
            path_msg.header = self._line_hdr if self._line_hdr else (self._drum_hdr or None)
            for x, y, z in center_strip:
                ps = PoseStamped()
                ps.header = path_msg.header
                ps.pose.position.x = float(x)
                ps.pose.position.y = float(y)
                ps.pose.position.z = float(z)
                path_msg.poses.append(ps)
            self.pub_center_path.publish(path_msg)

    # -------- Helpers: strip building --------
    def _build_strip(self, pts_xyz, min_pts, sort_by, smooth_window, ds):
        """점들을 정렬 → (선택) 스무딩 → (선택) 리샘플해서 연결선 좌표 리스트 반환."""
        if len(pts_xyz) < min_pts:
            return []

        P = np.asarray(pts_xyz, dtype=float)  # (N,3)
        # 정렬 기준
        if sort_by == 'range':
            order_key = np.hypot(P[:,0], P[:,1])  # xy거리
        else:
            order_key = P[:,0]  # x(전방) 기준
        P = P[np.argsort(order_key)]

        # 스무딩(이동평균) - 창 길이>=3 홀수 권장
        if smooth_window >= 3 and smooth_window % 2 == 1 and len(P) >= smooth_window:
            P[:,0] = self._moving_avg(P[:,0], smooth_window)
            P[:,1] = self._moving_avg(P[:,1], smooth_window)
            # z는 그대로 두거나 원하면 스무딩 가능
            # P[:,2] = self._moving_avg(P[:,2], smooth_window)

        if ds is None or ds <= 0.0 or len(P) < 2:
            return [(float(x), float(y), float(z)) for x,y,z in P]

        # 리샘플(등간격 arclength)
        S = self._cum_arclength(P[:,:2])
        L = S[-1]
        if L < ds:
            return [(float(x), float(y), float(z)) for x,y,z in P]

        s_new = np.arange(0.0, L + 1e-6, ds)
        x_new = np.interp(s_new, S, P[:,0])
        y_new = np.interp(s_new, S, P[:,1])
        # z는 가까운 이웃 보간(간단히 1D interp로 대체)
        z_new = np.interp(s_new, S, P[:,2])
        return [(float(x), float(y), float(z)) for x, y, z in zip(x_new, y_new, z_new)]

    def _build_centerline(self, left_xyz, right_xyz, ds_center):
        """좌/우 스트립을 각자 arclength로 파라미터화하고,
        공통 s축(0~min(Ll,Lr))에 보간한 뒤 평균을 취해 센터라인 생성."""
        L = np.asarray(left_xyz, dtype=float)
        R = np.asarray(right_xyz, dtype=float)
        if len(L) < 2 or len(R) < 2:
            return []

        Sl = self._cum_arclength(L[:,:2])
        Sr = self._cum_arclength(R[:,:2])
        Ll, Lr = Sl[-1], Sr[-1]
        Lmin = min(Ll, Lr)
        if Lmin < ds_center:
            return []

        s_common = np.arange(0.0, Lmin + 1e-6, ds_center)

        xl = np.interp(s_common, Sl, L[:,0]); yl = np.interp(s_common, Sl, L[:,1]); zl = np.interp(s_common, Sl, L[:,2])
        xr = np.interp(s_common, Sr, R[:,0]); yr = np.interp(s_common, Sr, R[:,1]); zr = np.interp(s_common, Sr, R[:,2])

        xc = 0.5 * (xl + xr); yc = 0.5 * (yl + yr); zc = 0.5 * (zl + zr)
        return [(float(x), float(y), float(z)) for x, y, z in zip(xc, yc, zc)]

    # -------- Helpers: markers --------
    def _make_sphere_list_marker(self, header, ns, mid, points, scale, color: ColorRGBA):
        m = Marker()
        m.ns = ns; m.id = mid; m.type = Marker.SPHERE_LIST
        if header is not None:
            m.header = header
        if not points:
            m.action = Marker.DELETE
            return m
        m.action = Marker.ADD
        m.scale.x = scale; m.scale.y = scale; m.scale.z = scale
        m.color = color
        for x,y,z in points:
            p = Point(); p.x=float(x); p.y=float(y); p.z=float(z)
            m.points.append(p)
        return m

    def _make_line_strip_marker(self, header, ns, mid, points, width, color: ColorRGBA):
        m = Marker()
        m.ns = ns; m.id = mid; m.type = Marker.LINE_STRIP
        if header is not None:
            m.header = header
        if not points or len(points) < 2:
            m.action = Marker.DELETE
            return m
        m.action = Marker.ADD
        m.scale.x = float(width)   # LINE_STRIP는 scale.x만 사용
        m.color = color
        for x,y,z in points:
            p = Point(); p.x=float(x); p.y=float(y); p.z=float(z)
            m.points.append(p)
        return m

    # -------- Utils --------
    def _read_xyzr(self, msg: PointCloud2):
        pts = []
        try:
            for x, y, z, r in pc2.read_points(msg, field_names=('x', 'y', 'z', 'range'), skip_nans=True):
                pts.append((x, y, z, r))
        except Exception:
            for x, y, z in pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True):
                rng = math.sqrt(x*x + y*y + z*z)
                pts.append((x, y, z, rng))
        return pts

    def _as_color(self, seq):
        rgba = ColorRGBA()
        try:
            rgba.r = float(seq[0]); rgba.g = float(seq[1])
            rgba.b = float(seq[2]); rgba.a = float(seq[3])
        except Exception:
            rgba.r, rgba.g, rgba.b, rgba.a = 1.0, 1.0, 1.0, 1.0
        return rgba

    def _with_alpha(self, c: ColorRGBA, a: float):
        c2 = ColorRGBA()
        c2.r, c2.g, c2.b, c2.a = c.r, c.g, c.b, float(max(0.0, min(1.0, a)))
        return c2

    def _moving_avg(self, arr, k):
        # 가장 단순한 동일가중 이동평균; 양끝은 패딩-리플렉트로 처리
        pad = k // 2
        a = np.pad(arr, (pad, pad), mode='edge')
        kernel = np.ones(k, dtype=float) / k
        return np.convolve(a, kernel, mode='valid')

    def _cum_arclength(self, xy):
        # xy: (N,2), 인접 점 간 거리 누적
        d = np.sqrt(np.sum(np.diff(xy, axis=0)**2, axis=1))
        S = np.concatenate(([0.0], np.cumsum(d)))
        return S


def main(args=None):
    rclpy.init(args=args)
    node = LinePointsMarkers()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
