#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

def norm(frame: str) -> str:
    return frame.lstrip('/') if isinstance(frame, str) else frame

class Odom2TF(Node):
    def __init__(self):
        super().__init__('odom2tf')
        # 기본 파라미터 (오면 메시지 값이 우선)
        self.def_parent = self.declare_parameter('parent_frame', 'odom').get_parameter_value().string_value
        self.def_child  = self.declare_parameter('child_frame',  'base_link').get_parameter_value().string_value
        self.odom_topic = self.declare_parameter('odom_topic', '/kiss/odometry').get_parameter_value().string_value

        self.br = TransformBroadcaster(self)
        self.sub = self.create_subscription(Odometry, self.odom_topic, self.cb, 10)
        self.get_logger().info(f"[odom2tf] rebroadcast TF from {self.odom_topic}")

    def cb(self, msg: Odometry):
        parent = norm(msg.header.frame_id) or self.def_parent
        child  = norm(msg.child_frame_id)  or self.def_child

        t = TransformStamped()
        t.header.stamp = msg.header.stamp           # ★ 오돔의 stamp 사용(센서 시간과 정렬)
        t.header.frame_id = parent
        t.child_frame_id  = child
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation = msg.pose.pose.orientation
        self.br.sendTransform(t)

def main():
    rclpy.init()
    rclpy.spin(Odom2TF())
    rclpy.shutdown()

if __name__ == '__main__':
    main()