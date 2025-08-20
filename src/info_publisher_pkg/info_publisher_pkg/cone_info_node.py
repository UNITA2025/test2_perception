import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import numpy as np
import struct
from interfaces_pkg.msg import ConeInfo, ConeInfoArray


class ConeinfoNode(Node):
    def __init__(self):
        super().__init__('cone_info_node')

        # 토픽 매핑: 입력 → 출력
        self.topic_pairs = [
            ('/cones_left/colored_points_cone', '/cones/cone_info_left'),
            ('/cones_right/colored_points_cone', '/cones/cone_info_right'),
            ('/cones_down/colored_points_cone', '/cones/cone_info_down'),
        ]

        # 구독자와 퍼블리셔 생성
        self.subs = []
        self.pubs = {}
        for sub_topic, pub_topic in self.topic_pairs:
            self.subs.append(self.create_subscription(PointCloud2, sub_topic, self.make_callback(pub_topic), 10))
            self.pubs[pub_topic] = self.create_publisher(ConeInfoArray, pub_topic, 10)

        self.get_logger().info("ConeinfoNode started: Listening to cone topics and publishing cone info.")

    def make_callback(self, pub_topic):
        def callback(msg: PointCloud2):
            cloud = self.pc2_to_array(msg)
            cone_list = []
            for pt in cloud:
                x, y, z = pt[:3]
                rgb_f = pt[3]
                r, g, b = self.unpack_rgb_float(rgb_f)
                dist = float(np.linalg.norm([x, y, z]))

                cone = ConeInfo()
                cone.x = float(x)
                cone.y = float(y)
                cone.z = float(z)
                cone.distance = dist
                cone.cone_color = self.rgb_to_color_name(r, g, b)
                cone_list.append(cone)

            cone_list.sort(key=lambda cone: cone.distance)
            msg_out = ConeInfoArray()
            msg_out.cones = cone_list
            self.pubs[pub_topic].publish(msg_out)
        return callback

    def pc2_to_array(self, msg: PointCloud2):
        return np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 4)

    def unpack_rgb_float(self, rgb_float: float):
        rgb_int = struct.unpack('I', struct.pack('f', rgb_float))[0]
        return (rgb_int >> 16) & 0xFF, (rgb_int >> 8) & 0xFF, rgb_int & 0xFF

    def rgb_to_color_name(self, r, g, b):
        if b > r and b > g:
            return "blue"
        else:
            return "yellow"

def main(args=None):
    rclpy.init(args=args)
    node = ConeinfoNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
