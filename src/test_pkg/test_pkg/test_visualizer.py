#!/usr/bin/env python3
import math
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
        # 입력 토픽들
        self.declare_parameter('line_topic', '/line_points')
        self.declare_parameter('drum_topic', '/drum_points')
        # 출력 마커 토픽
        self.declare_parameter('output_topic', '/line_points_markers')

        # 라인(파란색) 점 마커 설정
        self.declare_parameter('line_point_scale', 0.10)                 # [m]
        self.declare_parameter('line_point_color', [0.2, 0.6, 1.0, 1.0]) # r,g,b,a (파랑)

        # 드럼(빨간색) 점 마커 설정
        self.declare_parameter('drum_point_scale', 0.12)                 # [m]
        self.declare_parameter('drum_point_color', [1.0, 0.2, 0.2, 1.0]) # r,g,b,a (빨강)

        # 과밀 시 샘플링(각 토픽별)
        self.declare_parameter('line_decimate', 1)   # N: N개마다 하나 사용(1=모두)
        self.declare_parameter('drum_decimate', 1)

        line_topic = self.get_parameter('line_topic').value
        drum_topic = self.get_parameter('drum_topic').value
        out_topic  = self.get_parameter('output_topic').value

        # 최신 포인트/헤더 저장
        self._line_pts = []   # [(x,y,z,range), ...]
        self._line_hdr = None
        self._drum_pts = []
        self._drum_hdr = None

        # 구독/퍼블리셔
        self.sub_line = self.create_subscription(PointCloud2, line_topic, self._line_cb, 10)
        self.sub_drum = self.create_subscription(PointCloud2, drum_topic, self._drum_cb, 10)
        self.pub      = self.create_publisher(MarkerArray, out_topic, 10)

        self.get_logger().info(
            f"✅ line_points_markers started (line: {line_topic}, drum: {drum_topic} → out: {out_topic})"
        )

    # -------- Callbacks --------
    def _line_cb(self, msg: PointCloud2):
        decimate = max(1, int(self.get_parameter('line_decimate').value))
        pts = self._read_xyzr(msg)
        if decimate > 1:
            pts = pts[::decimate]
        self._line_pts = pts
        self._line_hdr = msg.header
        self._publish_markers()

    def _drum_cb(self, msg: PointCloud2):
        decimate = max(1, int(self.get_parameter('drum_decimate').value))
        pts = self._read_xyzr(msg)
        if decimate > 1:
            pts = pts[::decimate]
        self._drum_pts = pts
        self._drum_hdr = msg.header
        self._publish_markers()

    # -------- Marker Publisher --------
    def _publish_markers(self):
        arr = MarkerArray()
        ns = "line_drum_points"

        # ---- Line marker (id=0, 파란색) ----
        m_line = Marker()
        m_line.ns = ns
        m_line.id = 0
        if self._line_hdr is not None:
            m_line.header = self._line_hdr
        m_line.type = Marker.SPHERE_LIST
        # 비어있으면 삭제
        if len(self._line_pts) == 0:
            m_line.action = Marker.DELETE
            arr.markers.append(m_line)
        else:
            m_line.action = Marker.ADD
            s = float(self.get_parameter('line_point_scale').value)
            m_line.scale.x = s; m_line.scale.y = s; m_line.scale.z = s
            m_line.color = self._as_color(self.get_parameter('line_point_color').value)
            for x, y, z, _ in self._line_pts:
                p = Point(); p.x = float(x); p.y = float(y); p.z = float(z)
                m_line.points.append(p)
            arr.markers.append(m_line)

        # ---- Drum marker (id=1, 빨간색) ----
        m_drum = Marker()
        m_drum.ns = ns
        m_drum.id = 1
        if self._drum_hdr is not None:
            m_drum.header = self._drum_hdr
        m_drum.type = Marker.SPHERE_LIST
        if len(self._drum_pts) == 0:
            m_drum.action = Marker.DELETE
            arr.markers.append(m_drum)
        else:
            m_drum.action = Marker.ADD
            s = float(self.get_parameter('drum_point_scale').value)
            m_drum.scale.x = s; m_drum.scale.y = s; m_drum.scale.z = s
            m_drum.color = self._as_color(self.get_parameter('drum_point_color').value)
            for x, y, z, _ in self._drum_pts:
                p = Point(); p.x = float(x); p.y = float(y); p.z = float(z)
                m_drum.points.append(p)
            arr.markers.append(m_drum)

        # 퍼블리시
        if arr.markers:
            self.pub.publish(arr)

    # -------- Utils --------
    def _read_xyzr(self, msg: PointCloud2):
        """(x,y,z,range) 필드가 있으면 그대로, 없으면 range를 계산해서 반환."""
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
