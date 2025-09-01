import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

from interfaces_pkg.msg import LaneInfo  # custom msg

class LanePathVisualizer(Node):
    def __init__(self):
        super().__init__('lane_path_visualizer_node')

        # Subscriber
        self.image_sub = self.create_subscription(
            Image, '/roi_image', self.image_callback, 10)
        self.laneinfo_sub = self.create_subscription(
            LaneInfo, '/yolov8_lane_info', self.laneinfo_callback, 10)

        # Publisher
        
        self.image_pub = self.create_publisher(Image, '/lane_path_image', 10)

        self.bridge = CvBridge()
        self.latest_image = None
        self.latest_laneinfo = None

    def image_callback(self, msg):
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.update_and_publish()

    def laneinfo_callback(self, msg):
        self.latest_laneinfo = msg
        self.update_and_publish()

    def update_and_publish(self):
        if self.latest_image is None or self.latest_laneinfo is None:
            return

        img = self.latest_image.copy()

        # Lane points 가져오기
        points = []
        for tp in self.latest_laneinfo.target_points:
            points.append((int(tp.target_x), int(tp.target_y)))

        # 이미지 위에 그리기
        for pt in points:
            cv2.circle(img, pt, 5, (0, 0, 255), -1)  # 빨간 점 표시

        if len(points) > 1:
            for i in range(len(points) - 1):
                cv2.line(img, points[i], points[i + 1], (0, 255, 0), 2)  # 초록 선 연결

        # ROS 퍼블리시
        msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
        self.image_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LanePathVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
