#!/usr/bin/env python3
import rclpy
import cv2
import numpy as np
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
from interfaces_pkg.msg import TrafficLightState
from collections import deque

### 수정된 부분: HSV_RANGE에 'left_turn' 추가 ###
# 좌회전은 초록색이므로 green과 동일한 범위를 사용합니다.
HSV_RANGE = {
    "red":       [((  0, 80, 120), ( 10, 255, 255)), ((160, 80, 120), (180, 255, 255))],
    "yellow":    [(( 20, 80, 120), ( 35, 255, 255))],
    "green":     [(( 75, 80, 120), ( 95, 255, 255))],
    "left_turn": [(( 75, 80, 120), ( 95, 255, 255))]
}
MIN_PIXEL_RATIO_THRESHOLD = 0.05

### 수정된 부분: 수평 및 4색 신호등을 위한 새로운 분석 함수 ###
def classify_by_region_horizontal(bgr_roi: np.ndarray) -> tuple[str, float]:
    """
    ROI를 수평(좌->우)으로 4개 구역으로 나누고, 각 영역의 색상 성분을 분석합니다.
    """
    if bgr_roi.size == 0:
        return "off", 0.0

    height, width, _ = bgr_roi.shape
    
    # 1. ROI를 수평으로 4등분하여 각 신호등의 위치를 정의
    w_4_1 = width // 4
    w_4_2 = w_4_1 * 2
    w_4_3 = w_4_1 * 3
    
    # 각 구역에 해당하는 신호 상태와 ROI를 매핑
    # 순서: 빨강 | 좌회전 | 노랑 | 초록
    regions = {
        "red":       bgr_roi[:, 0:w_4_1],
        "left_turn": bgr_roi[:, w_4_1:w_4_2],
        "yellow":    bgr_roi[:, w_4_2:w_4_3],
        "green":     bgr_roi[:, w_4_3:width]
    }
    
    scores = {}
    
    # 2. 각 영역을 순회하며 해당 영역에 맞는 색상 픽셀 수 계산
    for state, roi in regions.items():
        if roi.size == 0:
            scores[state] = 0
            continue
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        pixel_count = 0
        for hsv_range in HSV_RANGE[state]:
            lower, upper = hsv_range
            mask = cv2.inRange(hsv, lower, upper)
            pixel_count += cv2.countNonZero(mask)
        
        scores[state] = pixel_count
        
    # 3. 가장 높은 점수를 받은 상태를 현재 신호로 판단
    if not scores:
        return "off", 0.0
        
    best_state = max(scores, key=scores.get)
    best_score = scores[best_state]
    
    # 4. 활성화 임계값 체크
    roi_area = (width * height) / 4
    if roi_area > 0 and (best_score / roi_area) < MIN_PIXEL_RATIO_THRESHOLD:
        return "off", 0.0
        
    confidence = best_score / roi_area
    return best_state, confidence


class TLDetector(Node):
    def __init__(self):
        super().__init__("tl_detector")

        self.declare_parameter("image_topic", "camera_front_up/image_raw")
        self.declare_parameter("model_path",  "traffic_light.pt")
        self.declare_parameter("conf_th",     0.30)

        img_topic  = self.get_parameter("image_topic").get_parameter_value().string_value
        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        self.conf_th = self.get_parameter("conf_th").get_parameter_value().double_value

        self.bridge = CvBridge()
        self.model  = YOLO(model_path)

        self.sub = self.create_subscription(Image, img_topic,
                                            self.cb_img, qos_profile_sensor_data)
        self.pub = self.create_publisher(TrafficLightState, "/traffic_light_state", 10)
        self.dbg = self.create_publisher(Image, "/traffic_light_debug",10)
        
        self.HISTORY_SIZE = 15
        self.STABLE_THRESHOLD = 7
        self.state_history = deque(maxlen=self.HISTORY_SIZE)
        self.confirmed_state = "off"
        self.confirmed_confidence = 0.0

    def cb_img(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        drawn = frame.copy()
        
        all_boxes = self.model(frame, conf=self.conf_th, verbose=False)[0].boxes

        target_box = None
        max_area = -1
        
        if all_boxes:
            for box in all_boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    target_box = box
        
        raw_state = "off"
        raw_conf = 0.0
        
        if target_box is not None:
            x1, y1, x2, y2 = map(int, target_box.xyxy[0])
            roi = frame[y1:y2, x1:x2]
            
            ### 수정된 부분: 새로운 수평 분석 함수 호출 ###
            raw_state, raw_conf = classify_by_region_horizontal(roi)

            # 디버그 시각화 색상에 left_turn 추가 (시안 색상)
            color = (0, 255, 0)   if raw_state == "green" else \
                    (0, 255, 255) if raw_state == "yellow" else \
                    (0, 0, 255)   if raw_state == "red" else \
                    (255, 255, 0) if raw_state == "left_turn" else \
                    (200, 200, 200)
            cv2.rectangle(drawn, (x1, y1), (x2, y2), color, 2)
            cv2.putText(drawn, f"RAW: {raw_state} ({raw_conf:.2f})", (x1, y1 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        self.state_history.append(raw_state)

        if len(self.state_history) == self.HISTORY_SIZE:
            most_common_state = max(set(self.state_history), key=self.state_history.count)
            count = self.state_history.count(most_common_state)

            if count >= self.STABLE_THRESHOLD:
                self.confirmed_state = most_common_state
                self.confirmed_confidence = count / self.HISTORY_SIZE

        self.pub.publish(TrafficLightState(state=self.confirmed_state, 
                                           confidence=float(self.confirmed_confidence)))

        cv2.putText(drawn, f"CONFIRMED STATE: {self.confirmed_state.upper()}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
        cv2.putText(drawn, f"CONFIRMED STATE: {self.confirmed_state.upper()}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        self.dbg.publish(self.bridge.cv2_to_imgmsg(drawn, "bgr8"))

def main():
    rclpy.init()
    rclpy.spin(TLDetector())
    rclpy.shutdown()

if __name__ == "__main__":
    main()