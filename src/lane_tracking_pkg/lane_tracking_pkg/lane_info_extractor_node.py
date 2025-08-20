import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from interfaces_pkg.msg import TargetPoint, LaneInfo, DetectionArray, BoundingBox2D, Detection
from .lib import camera_perception_func_lib as CPFL

from interfaces_pkg.msg import Header

# ==== 새로 추가: CPFL 없이 ROI 직접 자르기 ====
import numpy as np

def make_roi(image, *, mode='middle',
             top_ratio=0.35, bottom_ratio=0.75,
             top_px=None, bottom_px=None):
    """
    image: HxW 또는 HxWxC (np.uint8/float 등 상관없음)
    mode:
      - 'below'  : bottom_px 또는 bottom_ratio 기준으로 하부 사용
      - 'above'  : top_px 또는 top_ratio 기준으로 상부 사용
      - 'middle' : [top, bottom] 사이의 중간 대역 사용 (행님이 원하는 모드)
    비율은 [0.0, 1.0] 범위. 픽셀이 주어지면 픽셀 우선.
    """
    h = image.shape[0]

    # 비율 → 픽셀로 환산
    t = top_px if top_px is not None else int(round(h * top_ratio))
    b = bottom_px if bottom_px is not None else int(round(h * bottom_ratio))

    # 모드별 y-범위 결정
    if mode == 'below':
        y0, y1 = b, h
    elif mode == 'above':
        y0, y1 = 0, t
    else:  # 'middle'
        # 보정(경계 체크 & 뒤집힘 방지)
        y0, y1 = min(max(t, 0), h), min(max(b, 0), h)
        if y0 > y1:
            y0, y1 = y1, y0

    # 슬라이스
    roi = image[y0:y1, :].copy()
    return roi, (y0, y1)

#---------------Variable Setting---------------
# Subscribe할 토픽 이름
SUB_TOPIC_NAME = "detections"

# Publish할 토픽 이름
PUB_TOPIC_NAME = "yolov8_lane_info"
ROI_IMAGE_TOPIC_NAME = "roi_image"  # 추가: ROI 이미지 퍼블리시 토픽

# 화면에 이미지를 처리하는 과정을 띄울것인지 여부: True, 또는 False 중 택1하여 입력
SHOW_IMAGE = False
#----------------------------------------------


class Yolov8InfoExtractor(Node):
    def __init__(self):
        super().__init__('lane_info_extractor_node')

        # === 튜닝 파라미터 (ROS param으로 노출) ===
        self.src_mat_param = self.declare_parameter(
            'src_mat', [238,316,  402,313,  501,476,  155,476]  # [x0,y0, x1,y1, x2,y2, x3,y3]
        ).value
        self.dst_left_ratio_top  = self.declare_parameter('dst_left_ratio_top', 0.405).value
        self.dst_right_ratio_top = self.declare_parameter('dst_right_ratio_top', 0.595).value
        self.dst_left_ratio_bot  = self.declare_parameter('dst_left_ratio_bot', 0.300).value
        self.dst_right_ratio_bot = self.declare_parameter('dst_right_ratio_bot',0.700).value

        self.roi_top_ratio    = self.declare_parameter('roi_top_ratio', 0.35).value
        self.roi_bottom_ratio = self.declare_parameter('roi_bottom_ratio', 0.75).value
        self.frame_id_bev     = self.declare_parameter('frame_id_bev', 'bev').value

        # 타겟 포인트 y 개수/간격
        self.num_target_rows  = self.declare_parameter('num_target_rows', 4).value  # 4줄
        self.row_margin_px    = self.declare_parameter('row_margin_px', 5).value    # 위/아래 여유

        # (QoS/Publisher/Subscribers 동일)

    def yolov8_detections_callback(self, detection_msg: DetectionArray):
        if len(detection_msg.detections) == 0:
            return

        # 1) 에지 이미지
        lane_edge_image = CPFL.draw_edges(detection_msg, cls_name='lane', color=255)
        h, w = lane_edge_image.shape[:2]

        # 2) src/dst 점 구성 (float32, 시계방향)
        src_vals = self.src_mat_param
        if len(src_vals) != 8:
            self.get_logger().warn("src_mat length must be 8; fallback to defaults.")
            src_vals = [238,316, 402,313, 501,476, 155,476]
        src_mat = np.array(src_vals, dtype=np.float32).reshape(4,2)

        dst_mat = np.array([
            [w * self.dst_left_ratio_top,  0.0],  # 좌상
            [w * self.dst_right_ratio_top, 0.0],  # 우상
            [w * self.dst_right_ratio_bot, h],    # 우하
            [w * self.dst_left_ratio_bot,  h],    # 좌하
        ], dtype=np.float32)

        # 3) BEV 변환
        lane_bird_image = CPFL.bird_convert(lane_edge_image, srcmat=src_mat.tolist(), dstmat=dst_mat.tolist())

        # 4) ROI 자르기 (중간 대역)
        roi_image, (y0, y1) = make_roi(
            lane_bird_image,
            mode='middle',
            top_ratio=float(self.roi_top_ratio),
            bottom_ratio=float(self.roi_bottom_ratio),
        )
        roi_h, roi_w = roi_image.shape[:2]

        # 5) 타입 정리 및 퍼블리시(헤더 포함)
        if roi_image.dtype != np.uint8:
            roi_image = cv2.convertScaleAbs(roi_image)

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.frame_id_bev

        try:
            roi_image_msg = self.cv_bridge.cv2_to_imgmsg(roi_image, encoding="mono8")
            roi_image_msg.header = header
            self.roi_image_publisher.publish(roi_image_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to convert/publish ROI image: {e}")

        # 6) dominant gradient
        grad = CPFL.dominant_gradient(roi_image, theta_limit=70)

        # 7) 타겟 포인트 y를 ROI 높이에 맞춰 자동 생성 (out-of-bounds 방지)
        rows = max(1, int(self.num_target_rows))
        top = self.row_margin_px
        bot = max(self.row_margin_px+1, roi_h - self.row_margin_px)
        ys = np.linspace(top, bot-1, rows).astype(int)

        target_points = []
        # (선택) 차선폭(m)→px로 치환하는 파라미터가 있다면 여기서 px로 환산
        lane_width_px = 300  # TODO: 파라미터화/환산

        for y in ys:
            x = CPFL.get_lane_center(
                roi_image,
                detection_height=int(y),
                detection_thickness=10,
                road_gradient=grad,
                lane_width=lane_width_px
            )
            tp = TargetPoint()
            # ROI 좌표로 줄지, BEV 전체 좌표로 줄지 결정
            # - ROI 좌표 유지: 그대로
            # - BEV 좌표로 변환: y += y0
            tp.target_x = int(round(x))
            tp.target_y = int(round(y))  # BEV 전체 좌표 필요하면 y + y0 로 변경
            target_points.append(tp)

        lane = LaneInfo()
        # lane.header가 있다면 꼭 채워주기
        if hasattr(lane, 'header'):
            lane.header = header
        lane.slope = grad
        lane.target_points = target_points

        self.publisher.publish(lane)

        if self.show_image:
            cv2.imshow('lane_edge_image', lane_edge_image)
            cv2.imshow('lane_bird_img', lane_bird_image)
            cv2.imshow('roi_img', roi_image)
            cv2.waitKey(1)
        # roi_image를 ROS Image 메시지로 변환
        try:
            roi_image_msg = self.cv_bridge.cv2_to_imgmsg(roi_image, encoding="mono8")
            # ROI 이미지를 퍼블리시
            self.roi_image_publisher.publish(roi_image_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to convert and publish ROI image: {e}")
        
        grad = CPFL.dominant_gradient(roi_image, theta_limit=70)
                
        target_points = []
        for target_point_y in range(5, 155, 50):  # 예시로 5에서 155까지 50씩 증가
            target_point_x = CPFL.get_lane_center(roi_image, detection_height=target_point_y, 
                                                detection_thickness=10, road_gradient=grad, lane_width=300)
            
            target_point = TargetPoint()
            target_point.target_x = round(target_point_x)
            target_point.target_y = round(target_point_y)
            target_points.append(target_point)

        lane = LaneInfo()
        lane.slope = grad
        lane.target_points = target_points

        self.publisher.publish(lane)


def main(args=None):
    rclpy.init(args=args)
    node = Yolov8InfoExtractor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()
  
if __name__ == '__main__':
    main()
