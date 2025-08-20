import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import numpy as np
import struct
from interfaces_pkg.msg import ConeInfo, ConeInfoArray

class DruminfoNode(Node):
    def __init__(self):
        super().__init__('drum_info_node')

        self.topic_pairs = [
            ('/cones_left/colored_points_drum', '/drums/drum_info_left'),
            ('/cones_right/colored_points_drum', '/drums/drum_info_right'),
            ('/cones_down/colored_points_drum', '/drums/drum_info_down'),
        ]

        self.subs = []
        self.pubs = {}
        for sub_topic, pub_topic in self.topic_pairs:
            self.subs.append(self.create_subscription(PointCloud2, sub_topic, self.make_callback(pub_topic), 10))
            self.pubs[pub_topic] = self.create_publisher(ConeInfoArray, pub_topic, 10)

        self.get_logger().info("DruminfoNode started: Listening to drum topics and publishing drum info.")

    def make_callback(self, pub_topic):
        def callback(msg: PointCloud2):
            cloud = self.pc2_to_array(msg)
            drum_list = []
            for pt in cloud:
                x, y, z = pt[:3]
                rgb_f = pt[3]
                r, g, b = self.unpack_rgb_float(rgb_f)
                dist = float(np.linalg.norm([x, y, z]))

                drum = ConeInfo()
                drum.x = float(x)
                drum.y = float(y)
                drum.z = float(z)
                drum.distance = dist
                drum.cone_color = "yellow"  # 드럼은 고정 색상
                drum_list.append(drum)

            drum_list.sort(key=lambda d: d.distance)
            msg_out = ConeInfoArray()
            msg_out.cones = drum_list
            self.pubs[pub_topic].publish(msg_out)
        return callback

    def pc2_to_array(self, msg: PointCloud2):
        return np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 4)

    def unpack_rgb_float(self, rgb_float: float):
        rgb_int = struct.unpack('I', struct.pack('f', rgb_float))[0]
        return (rgb_int >> 16) & 0xFF, (rgb_int >> 8) & 0xFF, rgb_int & 0xFF

def main(args=None):
    rclpy.init(args=args)
    node = DruminfoNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
