#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from interfaces_pkg.msg import ErpCmdMsg, ErpStatusMsg

class SimBridge(Node):
    def __init__(self):
        super().__init__('sim_erp_bridge')
        self.sub = self.create_subscription(ErpCmdMsg, '/erp42_ctrl_cmd', self.cb, 10)
        self.pub = self.create_publisher(ErpStatusMsg, '/erp42_status', 10)
        self.timer = self.create_timer(0.05, self.tick)  # 20 Hz
        self._cmd_speed = 0
        self._cmd_steer = 0
        self._alive = 0

    def cb(self, msg: ErpCmdMsg):
        self._cmd_speed = int(msg.speed)
        self._cmd_steer = int(msg.steer)

    def tick(self):
        out = ErpStatusMsg()
        out.control_mode = 1
        out.e_stop = False
        out.gear = 0
        out.speed = self._cmd_speed
        out.steer = self._cmd_steer
        out.brake = 0
        out.encoder = 0
        self._alive = (self._alive + 1) % 255
        out.alive = self._alive
        self.pub.publish(out)

def main():
    rclpy.init()
    n = SimBridge()
    rclpy.spin(n)
    n.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
