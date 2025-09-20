import cv2
import random
import numpy as np
from typing import Tuple

import rclpy
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn, LifecycleState
from rclpy.duration import Duration

import message_filters
from cv_bridge import CvBridge
from ultralytics.utils.plotting import Annotator, colors

from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
from interfaces_pkg.msg import BoundingBox2D, KeyPoint2D, KeyPoint3D, Detection, DetectionArray


class Yolov8VisualizerNode(LifecycleNode):
    def __init__(self) -> None:
        super().__init__('yolov8_visualizer_node')
        self._class_to_color = {}
        self.cv_bridge = CvBridge()
        self.declare_parameter('image_reliability', QoSReliabilityPolicy.RELIABLE)
        self.declare_parameter('image_topic', '')  # 반드시 런치 파일에서 지정
        self.declare_parameter('detections_topic', '')
        self.get_logger().info('Yolov8VisualizerNode created')

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f'Configuring {self.get_name()}')
        reliability = self.get_parameter('image_reliability').get_parameter_value().integer_value
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.detections_topic = self.get_parameter('detections_topic').get_parameter_value().string_value

        self.image_qos_profile = QoSProfile(
            reliability=reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        cam_name = self.get_name().replace("yolov8_visualizer_", "")
        self._dbg_pub = self.create_publisher(Image, f'/yolov8/{cam_name}/visualized_img', 10)
        self._bb_markers_pub = self.create_publisher(MarkerArray, f'/yolov8/{cam_name}/bb_markers', 10)
        self._kp_markers_pub = self.create_publisher(MarkerArray, f'/yolov8/{cam_name}/kp_markers', 10)

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f'Activating {self.get_name()}')
        img_sub = message_filters.Subscriber(
            self, Image, self.image_topic, qos_profile=self.image_qos_profile)
        det_sub = message_filters.Subscriber(
            self, DetectionArray, self.detections_topic, qos_profile=10)
        sync = message_filters.ApproximateTimeSynchronizer(
            [img_sub, det_sub], queue_size=10, slop=0.5)
        cam_name = self.get_name().replace("yolov8_visualizer_", "")
        sync.registerCallback(self._make_callback(cam_name))
        self._synchronizers = [sync]
        return TransitionCallbackReturn.SUCCESS

    def _make_callback(self, cam_name: str):
        def callback(img_msg: Image, detection_msg: DetectionArray) -> None:
            cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg)
            bb_markers = MarkerArray()
            kp_markers = MarkerArray()
            for det in detection_msg.detections:
                label = det.class_name
                if label not in self._class_to_color:
                    self._class_to_color[label] = (
                        random.randint(0,255),
                        random.randint(0,255),
                        random.randint(0,255))
                color = self._class_to_color[label]
                cv_image = self.draw_box(cv_image, det, color)
                cv_image = self.draw_mask(cv_image, det, color)
                cv_image = self.draw_keypoints(cv_image, det)
                if det.bbox3d.frame_id:
                    m = self.create_bb_marker(det, color)
                    m.header.stamp = img_msg.header.stamp
                    m.id = len(bb_markers.markers)
                    bb_markers.markers.append(m)
                if det.keypoints3d.frame_id:
                    for kp in det.keypoints3d.data:
                        mk = self.create_kp_marker(kp)
                        mk.header.frame_id = det.keypoints3d.frame_id
                        mk.header.stamp = img_msg.header.stamp
                        mk.id = len(kp_markers.markers)
                        kp_markers.markers.append(mk)
            dbg_img_msg = self.cv_bridge.cv2_to_imgmsg(cv_image, encoding=img_msg.encoding)
            self._dbg_pub.publish(dbg_img_msg)
            self._bb_markers_pub.publish(bb_markers)
            self._kp_markers_pub.publish(kp_markers)
        return callback

    # --- Existing drawing functions unchanged ---
    def draw_box(self, cv_image: np.ndarray, det: Detection, color: Tuple[int]) -> np.ndarray:
        box = det.bbox
        min_pt = (round(box.center.position.x - box.size.x/2),
                  round(box.center.position.y - box.size.y/2))
        max_pt = (round(box.center.position.x + box.size.x/2),
                  round(box.center.position.y + box.size.y/2))
        cv2.rectangle(cv_image, min_pt, max_pt, color, 2)
        label = f"{det.class_name} ({det.id}) ({det.score:.3f})"
        pos = (min_pt[0]+5, min_pt[1]+25)
        cv2.putText(cv_image, label, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    1, color, 1, cv2.LINE_AA)
        return cv_image

    def draw_mask(self, cv_image: np.ndarray, det: Detection, color: Tuple[int]) -> np.ndarray:
        mask = det.mask
        pts = np.array([[int(p.x), int(p.y)] for p in mask.data])
        if pts.size:
            layer = cv_image.copy()
            cv2.fillPoly(layer, [pts], color)
            cv2.addWeighted(cv_image, 0.4, layer, 0.6, 0, cv_image)
            cv2.polylines(cv_image, [pts], True, color, 2, cv2.LINE_AA)
        return cv_image

    def draw_keypoints(self, cv_image: np.ndarray, det: Detection) -> np.ndarray:
        ann = Annotator(cv_image)
        pts = det.keypoints.data
        for kp in pts:
            color_k = colors(kp.id-1) if len(pts)!=17 else ann.kpt_color[kp.id-1]
            cv2.circle(cv_image, (int(kp.point.x), int(kp.point.y)),
                       5, tuple(color_k), -1, cv2.LINE_AA)
        # draw skeleton
        for i, sk in enumerate(ann.skeleton):
            p1 = next((kp for kp in pts if kp.id==sk[0]), None)
            p2 = next((kp for kp in pts if kp.id==sk[1]), None)
            if p1 and p2:
                cv2.line(cv_image,
                         (int(p1.point.x),int(p1.point.y)),
                         (int(p2.point.x),int(p2.point.y)),
                         tuple(ann.limb_color[i]), 2, cv2.LINE_AA)
        return cv_image

    def create_bb_marker(self, det: Detection, color: Tuple[int]) -> Marker:
        obj = det.bbox3d
        m = Marker()
        m.header.frame_id = obj.frame_id
        m.ns = 'yolov8_3d'; m.type = Marker.CUBE; m.action = Marker.ADD
        m.pose.position = obj.center.position
        m.pose.orientation.w = 1.0
        m.scale.x, m.scale.y, m.scale.z = obj.size.x, obj.size.y, obj.size.z
        r, g, b = [c/255.0 for c in color]
        m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 0.4
        m.lifetime = Duration(seconds=0.5).to_msg(); m.text = det.class_name
        return m

    def create_kp_marker(self, kp3d: KeyPoint3D) -> Marker:
        m = Marker(); m.ns='yolov8_3d'; m.type=Marker.SPHERE; m.action=Marker.ADD
        m.pose.position = kp3d.point; m.pose.orientation.w=1.0
        m.scale.x=m.scale.y=m.scale.z=0.05
        m.color.r=(1-kp3d.score)*255/255; m.color.b=kp3d.score*255/255; m.color.a=0.4
        m.lifetime=Duration(seconds=0.5).to_msg(); m.text=str(kp3d.id)
        return m


def main():
    rclpy.init()
    node = Yolov8VisualizerNode()
    node.trigger_configure()
    node.trigger_activate()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
