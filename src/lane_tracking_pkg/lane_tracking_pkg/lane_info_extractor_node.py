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

from std_msgs.msg import Header

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
SUB_TOPIC_NAME = "/detections"

# Publish할 토픽 이름
PUB_TOPIC_NAME = "yolov8_lane_info"
ROI_IMAGE_TOPIC_NAME = "roi_image"  # 추가: ROI 이미지 퍼블리시 토픽

# 화면에 이미지를 처리하는 과정을 띄울것인지 여부: True, 또는 False 중 택1하여 입력
SHOW_IMAGE = False
#----------------------------------------------


from std_msgs.msg import Header

class Yolov8InfoExtractor(Node):
    def __init__(self):
        super().__init__('lane_info_extractor_node')

        # === 공통 파라미터 (아래 코드와 동일하게 꼭 존재시킬 것) ===
        self.sub_topic = self.declare_parameter('sub_detection_topic', SUB_TOPIC_NAME).value
        self.pub_topic = self.declare_parameter('pub_topic', PUB_TOPIC_NAME).value
        self.show_image = self.declare_parameter('show_image', SHOW_IMAGE).value

        # === 추가 파라미터화된 튜닝값 ===
        self.src_mat_param = self.declare_parameter(
            'src_mat', [238,316, 402,313, 501,476, 155,476]
        ).value
        self.dst_left_ratio_top  = float(self.declare_parameter('dst_left_ratio_top',  0.405).value)
        self.dst_right_ratio_top = float(self.declare_parameter('dst_right_ratio_top', 0.595).value)
        self.dst_left_ratio_bot  = float(self.declare_parameter('dst_left_ratio_bot',  0.300).value)
        self.dst_right_ratio_bot = float(self.declare_parameter('dst_right_ratio_bot', 0.700).value)

        self.roi_top_ratio    = float(self.declare_parameter('roi_top_ratio', 0.35).value)
        self.roi_bottom_ratio = float(self.declare_parameter('roi_bottom_ratio', 0.75).value)
        self.frame_id_bev     = self.declare_parameter('frame_id_bev', 'bev').value

        self.num_target_rows  = int(self.declare_parameter('num_target_rows', 4).value)
        self.row_margin_px    = int(self.declare_parameter('row_margin_px', 5).value)
        self.lane_width_px    = int(self.declare_parameter('lane_width_px', 300).value)

        # === 브릿지/QoS/퍼브/섭 초기화 (아래 코드와 '완전히' 동일하게 필수) ===
        self.cv_bridge = CvBridge()
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )
        self.subscriber = self.create_subscription(
            DetectionArray, self.sub_topic, self.yolov8_detections_callback, self.qos_profile
        )
        self.publisher = self.create_publisher(LaneInfo, self.pub_topic, self.qos_profile)
        self.roi_image_publisher = self.create_publisher(Image, ROI_IMAGE_TOPIC_NAME, self.qos_profile)

    def yolov8_detections_callback(self, detection_msg: DetectionArray):
        if not detection_msg.detections:
            return

        # 1) 에지
        lane_edge_image = CPFL.draw_edges(detection_msg, cls_name='lane', color=255)
        h, w = lane_edge_image.shape[:2]

        # 2) src/dst
        src_vals = self.src_mat_param if len(self.src_mat_param) == 8 else [238,316, 402,313, 501,476, 155,476]
        src_mat = np.array(src_vals, dtype=np.float32).reshape(4, 2)
        dst_mat = np.array([
            [w * self.dst_left_ratio_top,   0.0],
            [w * self.dst_right_ratio_top,  0.0],
            [w * self.dst_right_ratio_bot,  h],
            [w * self.dst_left_ratio_bot,   h],
        ], dtype=np.float32)

        # 3) BEV
        lane_bird_image = CPFL.bird_convert(lane_edge_image, srcmat=src_mat.tolist(), dstmat=dst_mat.tolist())

        # 4) ROI (비율 기반 안전가드)
        roi_image, (y0, y1) = make_roi(
            lane_bird_image,
            mode='middle',
            top_ratio=float(self.roi_top_ratio),
            bottom_ratio=float(self.roi_bottom_ratio),
        )

        if roi_image is None or roi_image.size == 0:
            self.get_logger().warn(
                f"ROI empty: top_ratio={self.roi_top_ratio}, bottom_ratio={self.roi_bottom_ratio}, h={lane_bird_image.shape[0]}"
            )
            return

        if roi_image.dtype != np.uint8:
            roi_image = cv2.convertScaleAbs(roi_image)

        # 5) ROI 퍼블리시 (헤더 포함)
        roi_header = Header()
        roi_header.stamp = self.get_clock().now().to_msg()
        roi_header.frame_id = self.frame_id_bev

        roi_msg = self.cv_bridge.cv2_to_imgmsg(roi_image, encoding="mono8")
        roi_msg.header = roi_header
        self.roi_image_publisher.publish(roi_msg)

        # 6) 기울기
        try:
            grad = CPFL.dominant_gradient(roi_image, theta_limit=70)
        except Exception as e:
            self.get_logger().error(f"dominant_gradient failed: {e}")
            return

        # 7) 타겟 Y 자동 생성 (경계 체크)
        top = self.row_margin_px
        bot = max(self.row_margin_px + 1, roi_image.shape[0] - self.row_margin_px)
        rows = max(1, int(self.num_target_rows))
        ys = np.linspace(top, bot - 1, rows).astype(int)

        target_points = []
        for y in ys:
            x = CPFL.get_lane_center(
                roi_image,
                detection_height=int(y),
                detection_thickness=10,
                road_gradient=float(grad),
                lane_width=int(self.lane_width_px),
            )
            tp = TargetPoint()
            tp.target_x = int(round(x))
            tp.target_y = int(round(y))     # 필요하면 y + y0 로 변경
            target_points.append(tp)

        # 8) LaneInfo 퍼블리시 (중복 제거, 한 번만)
        lane = LaneInfo()
        try:
            lane.header = roi_header
        except AttributeError:
            pass
        lane.slope = float(grad)
        lane.target_points = target_points
        self.publisher.publish(lane)

        if self.show_image:
             cv2.imshow('lane_edge_image', lane_edge_image)
             cv2.imshow('lane_bird_img', lane_bird_image)
             cv2.imshow('roi_img', roi_image)
             cv2.waitKey(1)


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
