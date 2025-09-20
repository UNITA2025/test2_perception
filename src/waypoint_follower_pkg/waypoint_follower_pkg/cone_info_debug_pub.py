#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from interfaces_pkg.msg import ConeInfo, ConeInfoArray

class ConeDebugPub(Node):
    def __init__(self):
        super().__init__('cone_info_debug_pub')
        self.pub = self.create_publisher(ConeInfoArray, '/cones/cone_info', 10)
        self.timer = self.create_timer(0.2, self.tick)  # 5 Hz

    def tick(self):
        msg = ConeInfoArray()
        def cone(x, y, color):
            c = ConeInfo(); c.x=x; c.y=y; c.z=0.0; c.distance=(x**2+y**2)**0.5; c.cone_color=color; return c
        msg.cones = [
            cone(2.0,  1.0, 'blue'),
            cone(4.0,  1.2, 'blue'),
            cone(2.0, -1.0, 'yellow'),
            cone(4.0, -1.2, 'yellow'),
        ]
        self.pub.publish(msg)

def main():
    rclpy.init(); n=ConeDebugPub(); rclpy.spin(n); n.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()
