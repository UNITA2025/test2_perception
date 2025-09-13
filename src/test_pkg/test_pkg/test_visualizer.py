#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs_py import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA


class LinePointsMarkers(Node):
    def __init__(self):
        super().__init__('line_points_markers')

        # -------- Parameters --------
        # 입력/출력 토픽
        self.declare_parameter('line_topic', '/line_points')
        self.declare_parameter('drum_topic', '/drum_points')
        self.declare_parameter('output_topic', '/line_points_markers')

        # 좌/우 분리 기준
        self.declare_parameter('y_left_is_neg', False)   # True면 y<0이 좌측
        self.declare_parameter('y_split_offset', 0.0)    # y 오프셋 (기준선 이동)
        self.declare_parameter('x_forward_min', 0.0)     # x<값 제거(뒤/근접 잡음 억제)
        self.declare_parameter('deadband_m', 0.7)        # 중앙 ±deadband: 내부 버림

        # 점(좌/우/드럼)
        self.declare_parameter('line_point_scale', 0.10)
        self.declare_parameter('line_left_color',  [0.2, 0.6, 1.0, 1.0])  # 파랑
        self.declare_parameter('line_right_color', [0.2, 1.0, 0.4, 1.0])  # 초록
        self.declare_parameter('drum_point_scale', 0.12)
        self.declare_parameter('drum_point_color', [1.0, 0.2, 0.2, 1.0])  # 빨강

        # 선(좌/우) 스타일
        self.declare_parameter('strip_width', 0.06)
        self.declare_parameter('strip_alpha', 1.0)

        # 스트립 빌드(정렬/스무딩/리샘플)
        self.declare_parameter('min_points_for_strip', 2)  # 이보다 적으면 선 안 그림
        self.declare_parameter('sort_by', 'x')             # 'x' or 'range'
        self.declare_parameter('smooth_window', 5)         # 홀수 권장, 1이면 미적용
        self.declare_parameter('resample_ds', 0.10)        # 등간격 간격[m], <=0 미적용

        # 과밀 시 샘플링
        self.declare_parameter('line_decimate', 1)
        self.declare_parameter('drum_decimate', 1)

        # ---- Near-field fill params (보간/외삽 표시) ----
        self.declare_parameter('fill_nearfield', True)   # 근거리 보간/외삽 on/off
        self.declare_parameter('fill_x_from', 0.5)       # 몇 m부터 채울지 (예: 0.5m)
        self.declare_parameter('fill_x_to', 2.0)         # 어디까지 채울지 (예: 2.0m)
        self.declare_parameter('fill_fit_deg', 2)        # polyfit 차수(2~3 권장)
        self.declare_parameter('fill_fit_min_pts', 5)    # 최소 피팅 포인트 수
        self.declare_parameter('fill_ds', 0.10)          # 보간 표출 간격(m)
        self.declare_parameter('fill_alpha', 0.55)       # 보간 표출 투명도

        # ----- Stable L/R memory & hysteresis -----
        self.declare_parameter('swap_hysteresis_m', 0.20)   # 스왑 전 요구되는 L/R 중심 y차
        self.declare_parameter('use_kmeans_when_ambiguous', True)  # 애매할 때 1D-2means 시도

        # 이전 프레임 좌/우 중심 (x,y), 없으면 None
        self._prev_left_c  = None
        self._prev_right_c = None

        # 파라미터 적용
        line_topic = self.get_parameter('line_topic').value
        drum_topic = self.get_parameter('drum_topic').value
        out_topic  = self.get_parameter('output_topic').value

        # 버퍼 (ndarray 사용)
        self._line_arr = None   # shape (N,4) = x,y,z,range
        self._line_hdr = None
        self._drum_arr = None
        self._drum_hdr = None

        # I/O
        self.sub_line = self.create_subscription(PointCloud2, line_topic, self._line_cb, 10)
        self.sub_drum = self.create_subscription(PointCloud2, drum_topic, self._drum_cb, 10)
        self.pub_markers = self.create_publisher(MarkerArray, out_topic, 10)

        self.get_logger().info(
            f"✅ line_points_markers (vectorized) started (line: {line_topic}, drum: {drum_topic} → out: {out_topic})"
        )

    # -------- Callbacks --------
    def _line_cb(self, msg: PointCloud2):
        decimate = max(1, int(self.get_parameter('line_decimate').value))
        arr = self._read_xyzr_numpy(msg)          # (N,4) ndarray
        if arr is not None and decimate > 1:
            arr = arr[::decimate]
        self._line_arr = arr
        self._line_hdr = msg.header
        self._publish_all()

    def _drum_cb(self, msg: PointCloud2):
        decimate = max(1, int(self.get_parameter('drum_decimate').value))
        arr = self._read_xyzr_numpy(msg)
        if arr is not None and decimate > 1:
            arr = arr[::decimate]
        self._drum_arr = arr
        self._drum_hdr = msg.header
        self._publish_all()

    # -------- Main Publisher --------
    def _publish_all(self):
        arr_msg = MarkerArray()
        ns = "line_drum_points"

        # 파라미터 로컬 캐싱 (함수 내 반복 호출 방지)
        y_left_is_neg = bool(self.get_parameter('y_left_is_neg').value)
        y_off         = float(self.get_parameter('y_split_offset').value)
        x_forward_min = float(self.get_parameter('x_forward_min').value)
        deadband      = float(self.get_parameter('deadband_m').value)

        line_point_scale = float(self.get_parameter('line_point_scale').value)
        drum_point_scale = float(self.get_parameter('drum_point_scale').value)
        line_left_color  = self._as_color(self.get_parameter('line_left_color').value)
        line_right_color = self._as_color(self.get_parameter('line_right_color').value)
        drum_point_color = self._as_color(self.get_parameter('drum_point_color').value)

        min_pts = int(self.get_parameter('min_points_for_strip').value)
        strip_w = float(self.get_parameter('strip_width').value)
        strip_alpha = float(self.get_parameter('strip_alpha').value)
        sort_by = str(self.get_parameter('sort_by').value).lower()
        smooth_window = int(self.get_parameter('smooth_window').value)
        ds = float(self.get_parameter('resample_ds').value)

        # ====== 좌/우 안정 분리 ======
        L_xyz, R_xyz = self._split_with_deadband_and_memory_vec(
            self._line_arr, y_left_is_neg, y_off, x_forward_min, deadband
        )  # ndarray (M,3) / (K,3)

        # ====== 점 마커 ======
        arr_msg.markers.append(self._make_sphere_list_marker_nd(
            header=self._line_hdr, ns=ns, mid=0, points=L_xyz,
            scale=line_point_scale, color=line_left_color
        ))
        arr_msg.markers.append(self._make_sphere_list_marker_nd(
            header=self._line_hdr, ns=ns, mid=1, points=R_xyz,
            scale=line_point_scale, color=line_right_color
        ))


        D_xyz = self._drum_arr[:, :3] if self._drum_arr is not None and self._drum_arr.size else np.empty((0,3))
        arr_msg.markers.append(self._make_sphere_list_marker_nd(
            header=self._drum_hdr, ns=ns, mid=2, points=D_xyz,
            scale=drum_point_scale, color=drum_point_color
        ))

        # ====== 라인 스트립 (정렬/스무딩/리샘플: 전부 NumPy) ======
        left_strip  = self._build_strip_vec(L_xyz, min_pts, sort_by, smooth_window, ds)
        right_strip = self._build_strip_vec(R_xyz, min_pts, sort_by, smooth_window, ds)

        arr_msg.markers.append(self._make_line_strip_marker_nd(
            header=self._line_hdr, ns=ns, mid=10, points=left_strip,
            width=strip_w, color=self._with_alpha(line_left_color, strip_alpha)
        ))
        arr_msg.markers.append(self._make_line_strip_marker_nd(
            header=self._line_hdr, ns=ns, mid=11, points=right_strip,
            width=strip_w, color=self._with_alpha(line_right_color, strip_alpha)
        ))


        # ====== 근거리 보간/외삽 (점선) ======
        if bool(self.get_parameter('fill_nearfield').value):
            x_from    = float(self.get_parameter('fill_x_from').value)
            x_to      = float(self.get_parameter('fill_x_to').value)
            fit_deg   = int(self.get_parameter('fill_fit_deg').value)
            fit_min   = int(self.get_parameter('fill_fit_min_pts').value)
            fill_ds   = float(self.get_parameter('fill_ds').value)
            fill_alpha= float(self.get_parameter('fill_alpha').value)

            left_fill  = self._fit_and_extrapolate_vec(left_strip,  x_from, x_to, fit_deg, fit_min, fill_ds)
            right_fill = self._fit_and_extrapolate_vec(right_strip, x_from, x_to, fit_deg, fit_min, fill_ds)

            arr_msg.markers.append(self._make_sphere_list_marker_nd(
                header=self._line_hdr, ns=ns, mid=12, points=left_fill,
                scale=line_point_scale*0.9, color=self._with_alpha(line_left_color, fill_alpha)
            ))
            arr_msg.markers.append(self._make_sphere_list_marker_nd(
                header=self._line_hdr, ns=ns, mid=13, points=right_fill,
                scale=line_point_scale*0.9, color=self._with_alpha(line_right_color, fill_alpha)
            ))

        if arr_msg.markers:
            self.pub_markers.publish(arr_msg)

    # -------- Stable split (fully vectorized) --------
    def _split_with_deadband_and_memory_vec(self, xyzi, y_left_is_neg, y_off, x_forward_min, deadband):
        """
        입력 xyzi: (N,4) ndarray or None
        반환: L_xyz, R_xyz (각각 (M,3)/(K,3) ndarray)
        """
        if xyzi is None or xyzi.size == 0:
            return np.empty((0,3)), np.empty((0,3))

        P = xyzi[xyzi[:,0] >= x_forward_min, :3]  # (x,y,z)
        if P.size == 0:
            return np.empty((0,3)), np.empty((0,3))

        y_adj = P[:,1] - y_off
        neg_mask = (y_adj <= -abs(deadband))
        pos_mask = (y_adj >=  abs(deadband))

        if y_left_is_neg:
            L0 = P[neg_mask]
            R0 = P[pos_mask]
        else:
            L0 = P[pos_mask]
            R0 = P[neg_mask]

        # 애매하면 1D-2means (y) 시도
        if ((L0.size == 0) or (R0.size == 0)) and bool(self.get_parameter('use_kmeans_when_ambiguous').value) and P.shape[0] >= 6:
            labels, centers_y = self._kmeans1d_y(P[:,1], k=2)
            idx_small = np.where(labels == np.argmin(centers_y))[0]
            idx_large = np.where(labels == np.argmax(centers_y))[0]
            if y_left_is_neg:
                L0 = P[idx_small]
                R0 = P[idx_large]
            else:
                L0 = P[idx_large]
                R0 = P[idx_small]

        # 이전 프레임 기반 스왑 방지 (히스테리시스)
        Lc_now = self._centroid_xy_nd(L0)
        Rc_now = self._centroid_xy_nd(R0)
        hyst   = float(self.get_parameter('swap_hysteresis_m').value)

        if (Lc_now is not None and Rc_now is not None and
            self._prev_left_c is not None and self._prev_right_c is not None):

            A = np.array(Lc_now); B = np.array(Rc_now)
            Lp = np.array(self._prev_left_c); Rp = np.array(self._prev_right_c)

            cost_1 = np.linalg.norm(A - Lp) + np.linalg.norm(B - Rp)
            cost_2 = np.linalg.norm(A - Rp) + np.linalg.norm(B - Lp)

            if cost_2 + 1e-6 < cost_1 and abs(A[1] - B[1]) > hyst:
                L0, R0 = R0, L0
                Lc_now, Rc_now = Rc_now, Lc_now

        elif Lc_now is not None and Rc_now is not None:
            # 초기 메모리 없는 경우: y순서 교정
            if abs(Lc_now[1] - Rc_now[1]) > hyst:
                if y_left_is_neg and (Lc_now[1] > Rc_now[1]):
                    L0, R0 = R0, L0
                    Lc_now, Rc_now = Rc_now, Lc_now
                elif (not y_left_is_neg) and (Lc_now[1] < Rc_now[1]):
                    L0, R0 = R0, L0
                    Lc_now, Rc_now = Rc_now, Lc_now

        else:
            # 한쪽만 있을 때 메모리로 추정
            if self._prev_left_c is not None and self._prev_right_c is not None:
                only = L0 if L0.size else R0
                if only.size:
                    Cnow = self._centroid_xy_nd(only)
                    dL = np.linalg.norm(np.array(Cnow) - np.array(self._prev_left_c))
                    dR = np.linalg.norm(np.array(Cnow) - np.array(self._prev_right_c))
                    if L0.size and (dR + hyst < dL):   # 현재 L만 있는데 R에 더 가까움 → 스왑
                        R0, L0 = L0, np.empty((0,3))
                    elif R0.size and (dL + hyst < dR): # 현재 R만 있는데 L에 더 가까움 → 스왑
                        L0, R0 = R0, np.empty((0,3))

        # 메모리 업데이트
        self._prev_left_c  = self._centroid_xy_nd(L0)
        self._prev_right_c = self._centroid_xy_nd(R0)

        return L0 if L0.size else np.empty((0,3)), R0 if R0.size else np.empty((0,3))

    # -------- Strip builder (vectorized) --------
    def _build_strip_vec(self, P, min_pts, sort_by, smooth_window, ds):
        """P: (N,3) ndarray → 등간격 아크길이 리샘플된 (M,3) ndarray"""
        if P is None or P.shape[0] < min_pts:
            return np.empty((0,3))
        # 정렬
        if sort_by == 'range':
            key = np.hypot(P[:,0], P[:,1])
        else:
            key = P[:,0]
        P = P[np.argsort(key)]

        # 스무딩 (이동평균, 가장자리 고정 패딩)
        if smooth_window >= 3 and smooth_window % 2 == 1 and P.shape[0] >= smooth_window:
            P = P.copy()
            P[:,0] = self._moving_avg_vec(P[:,0], smooth_window)
            P[:,1] = self._moving_avg_vec(P[:,1], smooth_window)
            # P[:,2] = self._moving_avg_vec(P[:,2], smooth_window)

        if ds is None or ds <= 0.0 or P.shape[0] < 2:
            return P

        # 등간격 아크길이 리샘플
        d = np.sqrt(np.sum(np.diff(P[:,:2], axis=0)**2, axis=1))
        S = np.concatenate(([0.0], np.cumsum(d)))
        Ltot = S[-1]
        if Ltot < ds:
            return P
        s_new = np.arange(0.0, Ltot + 1e-6, ds)
        x_new = np.interp(s_new, S, P[:,0])
        y_new = np.interp(s_new, S, P[:,1])
        z_new = np.interp(s_new, S, P[:,2])
        return np.stack([x_new, y_new, z_new], axis=1)

    # -------- Near-field fit/extrapolation (vectorized) --------
    def _fit_and_extrapolate_vec(self, P, x_from, x_to, deg, min_pts, ds):
        if P is None or P.shape[0] < max(min_pts, int(deg)+1):
            return np.empty((0,3))
        if np.ptp(P[:,0]) < 1e-3:
            return np.empty((0,3))
        coef = np.polyfit(P[:,0], P[:,1], deg=int(deg))
        poly = np.poly1d(coef)
        if x_from > x_to:
            x_from, x_to = x_to, x_from
        xs = np.arange(x_from, x_to + 1e-6, max(ds, 1e-3))
        if xs.size == 0:
            return np.empty((0,3))
        ys = poly(xs)
        z0 = float(P[0,2]) if P.shape[0] else 0.0
        zs = np.full_like(xs, z0, dtype=float)
        return np.stack([xs, ys, zs], axis=1)

    # -------- Marker builders (ndarray → list[Point]) --------
    def _make_sphere_list_marker_nd(self, header, ns, mid, points, scale, color: ColorRGBA):
        m = Marker()
        m.ns = ns; m.id = mid; m.type = Marker.SPHERE_LIST
        if header is not None:
            m.header = header
        if points is None or points.size == 0:
            m.action = Marker.DELETE
            return m
        m.action = Marker.ADD
        m.scale.x = scale; m.scale.y = scale; m.scale.z = scale
        m.color = color
        # list comprehension: CPython에서 append 루프보다 빠름
        m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in points]
        return m

    def _make_line_strip_marker_nd(self, header, ns, mid, points, width, color: ColorRGBA):
        m = Marker()
        m.ns = ns; m.id = mid; m.type = Marker.LINE_STRIP
        if header is not None:
            m.header = header
        if points is None or points.shape[0] < 2:
            m.action = Marker.DELETE
            return m
        m.action = Marker.ADD
        m.scale.x = float(width)
        m.color = color
        m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in points]
        return m

    # -------- Utils --------
    def _read_xyzr_numpy(self, msg: PointCloud2):
        """
        (우선) read_points_numpy 사용, 실패 시 fallback.
        반환: (N,4) ndarray [x,y,z,range] or None
        """
        try:
            # 최신 sensor_msgs_py에 존재. dtype: [('x','<f4'),('y','<f4'),('z','<f4'),('range','<f4')] 등
            arr = pc2.read_points_numpy(msg, field_names=('x','y','z','range'))
            if arr is None or arr.size == 0:
                return None
            # 일부 환경에서 structured array일 수 있으므로 일반 ndarray로 변환
            if arr.dtype.fields is not None:
                arr = np.stack([arr['x'], arr['y'], arr['z'], arr['range']], axis=1)
            return arr.astype(float, copy=False)
        except Exception:
            # Fallback: x,y,z만 읽고 3D range 계산
            try:
                arr3 = np.array(list(pc2.read_points(msg, field_names=('x','y','z'), skip_nans=True)), dtype=float)
                if arr3.size == 0:
                    return None
                rng = np.linalg.norm(arr3, axis=1, keepdims=True)
                return np.concatenate([arr3, rng], axis=1)
            except Exception:
                return None

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

    def _moving_avg_vec(self, arr, k):
        pad = k // 2
        a = np.pad(arr, (pad, pad), mode='edge')
        kernel = np.ones(k, dtype=float) / k
        return np.convolve(a, kernel, mode='valid')

    def _centroid_xy_nd(self, P):
        if P is None or P.size == 0:
            return None
        return (float(np.mean(P[:,0])), float(np.mean(P[:,1])))

    def _kmeans1d_y(self, y_vals, k=2, iters=15):
        y = np.asarray(y_vals, dtype=float).reshape(-1, 1)
        # 초기 중심: 분위수
        q = np.linspace(0.0, 1.0, k+2)[1:-1]
        centers = np.array([float(np.quantile(y, qi)) for qi in q], dtype=float).reshape(k, 1)

        for _ in range(iters):
            d = np.abs(y - centers.T)          # (N,k)
            labels = np.argmin(d, axis=1)
            new_centers = np.empty_like(centers)
            for j in range(k):
                idx = (labels == j)
                if not np.any(idx):
                    new_centers[j,0] = centers[j,0]
                else:
                    new_centers[j,0] = float(np.mean(y[idx]))
            if np.allclose(new_centers, centers):
                centers = new_centers
                break
            centers = new_centers
        return labels.astype(np.int32), centers.flatten().tolist()


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
