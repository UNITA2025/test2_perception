#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#==================================================#
# 기능(Function)
# - /lane_points, /drum_points 포인트클라우드( PointCloud2 )를 구독
# - 각각의 점을 MarkerArray(SPHERE_LIST)로 변환하여 RViz에 시각화
#   · lane → 파란색 점
#   · drum → 빨간색 점
#
# 노드(Node)
# - 이름: simple_points_visualizer
#
# 수신 토픽(Subscribe)
# - /lane_points (sensor_msgs/PointCloud2)
#   차선 후보 포인트 클라우드
# - /drum_points (sensor_msgs/PointCloud2)
#   드럼/장애물 후보 포인트 클라우드
#
# 송신 토픽(Publish)
# - /points_markers (visualization_msgs/MarkerArray)
#   line/drum 점들을 Marker(SPHERE_LIST)로 변환한 시각화용 데이터
#
# 파라미터(Parameters)
# - line_topic (str): 차선 입력 토픽 (기본: /lane_points)
# - drum_topic (str): 드럼 입력 토픽 (기본: /drum_points)
# - output_topic (str): 마커 출력 토픽 (기본: /points_markers)
#
# 처리 파이프라인(Flow)
# 1) PointCloud2 구독
# 2) (x,y,z) 좌표를 numpy 배열로 변환
# 3) SPHERE_LIST 마커 생성 (색상·스케일 적용)
# 4) MarkerArray로 묶어서 퍼블리시
#
# 참고(Notes)
# - RViz에서 MarkerArray 디스플레이 추가 후 /points_markers 선택하면 확인 가능
#
# TODO :
# 최종 수정일: 2025.09.19
# 편집자: 이기현, 정선우
#==================================================#

import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs_py import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

# ========= 기본 토픽 이름 =========
INPUT_LINE_TOPIC = "/lane_points"
INPUT_DRUM_TOPIC = "/drum_points"

OUTPUT_MARKER_TOPIC = "points_markers"


class SimplePointsVisualizer(Node):
    def __init__(self):
        super().__init__('simple_points_visualizer')

        # 파라미터: 입력/출력 토픽
        self.declare_parameter('line_topic', INPUT_LINE_TOPIC)
        self.declare_parameter('drum_topic', INPUT_DRUM_TOPIC)
        self.declare_parameter('output_topic', OUTPUT_MARKER_TOPIC)

        line_topic = self.get_parameter('line_topic').value
        drum_topic = self.get_parameter('drum_topic').value
        out_topic  = self.get_parameter('output_topic').value

        # 버퍼
        self._line_arr = None
        self._line_hdr = None
        self._drum_arr = None
        self._drum_hdr = None

        # I/O
        self.sub_line = self.create_subscription(PointCloud2, line_topic, self._line_cb, 10)
        self.sub_drum = self.create_subscription(PointCloud2, drum_topic, self._drum_cb, 10)
        self.pub_markers = self.create_publisher(MarkerArray, out_topic, 10)

        self.get_logger().info(f"✅ simple_points_visualizer started "
                               f"(line: {line_topic}, drum: {drum_topic} → out: {out_topic})")

    # -------- 콜백 --------
    def _line_cb(self, msg: PointCloud2):
        self._line_arr = self._read_xyz_numpy(msg)
        self._line_hdr = msg.header
        self._publish()

    def _drum_cb(self, msg: PointCloud2):
        self._drum_arr = self._read_xyz_numpy(msg)
        self._drum_hdr = msg.header
        self._publish()

    # -------- 발행 --------
    def _publish(self):
        arr_msg = MarkerArray()
        ns = "line_drum_points"

        # 색상/크기
        line_color = self._color_rgba(0.0, 0.0, 1.0, 1.0)   # 파랑
        drum_color = self._color_rgba(1.0, 0.0, 0.0, 1.0)   # 빨강
        scale = 0.12

        # line points
        if self._line_arr is not None and self._line_arr.size:
            arr_msg.markers.append(self._make_sphere_list_marker(
                header=self._line_hdr, ns=ns, mid=0,
                points=self._line_arr[:, :3], scale=scale, color=line_color
            ))

        # drum points
        if self._drum_arr is not None and self._drum_arr.size:
            arr_msg.markers.append(self._make_sphere_list_marker(
                header=self._drum_hdr, ns=ns, mid=1,
                points=self._drum_arr[:, :3], scale=scale, color=drum_color
            ))

        if arr_msg.markers:
            self.pub_markers.publish(arr_msg)

    # -------- Helpers --------
    def _make_sphere_list_marker(self, header, ns, mid, points, scale, color):
        m = Marker()
        m.ns = ns; m.id = mid; m.type = Marker.SPHERE_LIST
        m.header = header
        m.action = Marker.ADD
        m.scale.x = scale; m.scale.y = scale; m.scale.z = scale
        m.color = color
        m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in points]
        return m

    def _read_xyz_numpy(self, msg: PointCloud2):
        try:
            arr = pc2.read_points_numpy(msg, field_names=('x','y','z'))
            if arr is None or arr.size == 0:
                return None
            if arr.dtype.fields is not None:  # structured array
                arr = np.stack([arr['x'], arr['y'], arr['z']], axis=1)
            return arr.astype(float, copy=False)
        except Exception:
            try:
                arr = np.array(list(pc2.read_points(msg, field_names=('x','y','z'), skip_nans=True)), dtype=float)
                return arr if arr.size else None
            except Exception:
                return None

    def _color_rgba(self, r, g, b, a):
        c = ColorRGBA()
        c.r, c.g, c.b, c.a = float(r), float(g), float(b), float(a)
        return c


def main(args=None):
    rclpy.init(args=args)
    node = SimplePointsVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
