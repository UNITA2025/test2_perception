#!/usr/bin/env python3
import rclpy, cv2, sys
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class VideoPublisher(Node):
    def __init__(self):
        super().__init__("video_pub")  # ← launch 파일의 Node name과 동일하게

        # ── 런치/CLI 파라미터 ──
        self.declare_parameter("video_path", "demo.mp4")
        self.declare_parameter("fps", 30.0)

        path = self.get_parameter("video_path").get_parameter_value().string_value
        fps  = self.get_parameter("fps").get_parameter_value().double_value

        print(f"[INFO] Opening video file: {path}")
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            self.get_logger().error(f"[❌ ERROR] Cannot open video file: {path}")
            sys.exit(1)  # 명시적 종료

        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, "/image_raw", 10)
        self.timer = self.create_timer(1.0 / fps, self.timer_cb)

    def timer_cb(self):
        ret, frame = self.cap.read()
        if not ret:  # 영상 끝 → 루프 재시작
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return
        msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
        self.pub.publish(msg)

def main():
    rclpy.init()
    rclpy.spin(VideoPublisher())

if __name__ == "__main__":
    main()
