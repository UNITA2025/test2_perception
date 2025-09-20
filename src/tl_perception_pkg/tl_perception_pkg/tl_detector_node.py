#!/usr/bin/env python3
import rclpy
import cv2
import numpy as np
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
from interfaces_pkg.msg import TrafficLightState # 사용자 정의 메시지, 필요시 수정
from collections import deque

class TLDetector(Node):
    def __init__(self):
        super().__init__("tl_detector")

        # 파라미터 선언
        self.declare_parameter("image_topic", "camera_front_up/image_raw")
        self.declare_parameter("model_path",  "traffic_light_yolo_test.pt")
        self.declare_parameter("conf_th",     0.5) # 신뢰도 임계값, 필요시 조절

        # 파라미터 가져오기
        img_topic  = self.get_parameter("image_topic").get_parameter_value().string_value
        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        self.conf_th = self.get_parameter("conf_th").get_parameter_value().double_value

        # CV Bridge 및 YOLO 모델 초기화
        self.bridge = CvBridge()
        self.model  = YOLO(model_path)
        # 클래스 이름 확인 (예: ['green', 'red', 'yellow'])
        self.get_logger().info(f"Model classes: {self.model.names}") 
        
        # Subscriber, Publisher 설정
        self.sub = self.create_subscription(Image, img_topic, self.cb_img, qos_profile_sensor_data)
        self.pub = self.create_publisher(TrafficLightState, "/traffic_light_state", 10)
        self.dbg = self.create_publisher(Image, "/traffic_light_debug", 10)
        
        # 신호 안정화를 위한 설정
        self.HISTORY_SIZE = 15      # 몇 개의 프레임을 기억할지
        self.STABLE_THRESHOLD = 7   # history_size 중 이 횟수 이상 같은 신호가 나오면 확정
        self.state_history = deque(maxlen=self.HISTORY_SIZE)
        self.confirmed_state = "off"
        self.confirmed_confidence = 0.0

    def cb_img(self, msg: Image):
        # ROS Image 메시지를 OpenCV 이미지로 변환
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        drawn = frame.copy()
        
        # YOLOv8 모델로 객체 탐지 수행
        all_boxes = self.model(frame, conf=self.conf_th, verbose=False)[0].boxes

        target_box = None
        max_area = -1
        
        # 탐지된 박스들 중 가장 큰 박스를 찾음 (가장 가까운 신호등으로 가정)
        if all_boxes:
            for box in all_boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    target_box = box
        
        raw_state = "off"
        
        # 가장 큰 박스를 찾았으면, 그 박스의 클래스를 현재 신호로 판단
        if target_box is not None:
            # 클래스 ID를 가져옴 (예: 0, 1, 2)
            class_id = int(target_box.cls[0])
            # 모델에 저장된 클래스 이름 리스트에서 해당 ID의 이름을 가져옴
            raw_state = self.model.names[class_id]
            
            # --- 디버그 이미지에 결과 그리기 ---
            x1, y1, x2, y2 = map(int, target_box.xyxy[0])
            conf = float(target_box.conf[0])
            
            color = (0, 255, 0)  if "green" in raw_state else \
                    (0, 255, 255) if "yellow" in raw_state else \
                    (0, 0, 255)  if "red" in raw_state else \
                    (200, 200, 200)
                    
            cv2.rectangle(drawn, (x1, y1), (x2, y2), color, 2)
            cv2.putText(drawn, f"RAW: {raw_state} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 안정적인 신호 판단을 위해 history에 현재 상태 추가
        self.state_history.append(raw_state)

        # history가 꽉 찼을 때만 최종 신호 판단
        if len(self.state_history) == self.HISTORY_SIZE:
            # 가장 빈번하게 나타난 신호를 찾음
            most_common_state = max(set(self.state_history), key=self.state_history.count)
            count = self.state_history.count(most_common_state)

            # 일정 횟수 이상 나타났다면 최종 신호로 확정
            if count >= self.STABLE_THRESHOLD:
                self.confirmed_state = most_common_state
                self.confirmed_confidence = count / self.HISTORY_SIZE

        # 최종 확정된 신호 상태 발행
        self.pub.publish(TrafficLightState(state=self.confirmed_state,  
                                           confidence=float(self.confirmed_confidence)))

        # 화면에 최종 상태 표시
        cv2.putText(drawn, f"CONFIRMED STATE: {self.confirmed_state.upper()}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
        cv2.putText(drawn, f"CONFIRMED STATE: {self.confirmed_state.upper()}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # 디버그 이미지를 발행
        self.dbg.publish(self.bridge.cv2_to_imgmsg(drawn, "bgr8"))

def main():
    rclpy.init()
    node = TLDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()