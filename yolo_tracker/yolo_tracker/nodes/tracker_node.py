import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import message_filters

from vision_interfaces.msg import BoundingBox, BoundingBoxes 

from yolo_tracker.tracker.byte_track.byte_tracker import BYTETracker
from yolo_tracker.tracker.bot_sort.mc_bot_sort import BoTSORT

class TrackerArgs:
    def __init__(self,
                 # Common params
                 track_thresh=0.5, match_thresh=0.8,
                 track_buffer=30, mot20=False,
                 # BoT-SORT specific params
                 track_high_thresh=0.6, new_track_thresh=0.7,
                 proximity_thresh=0.5, appearance_thresh=0.25,
                 with_reid=False, fast_reid_config='', fast_reid_weights='',
                 device='cuda', cmc_method='sparseOptFlow', name='MOT17-02-FRCNN', ablation=False):

        # BYTETrack Params
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.mot20 = mot20

        # BoT-SORT Params
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = 0.1 # BoT-SORT는 low_thresh를 고정값으로 사용
        self.new_track_thresh = new_track_thresh
        self.track_buffer = track_buffer

        # ReID Params
        self.proximity_thresh = proximity_thresh
        self.appearance_thresh = appearance_thresh
        self.with_reid = with_reid
        self.fast_reid_config = fast_reid_config
        self.fast_reid_weights = fast_reid_weights
        self.device = device
        
        # GMC Params
        self.cmc_method = cmc_method
        self.name = name
        self.ablation = ablation


class TrackerNode(Node):
    def __init__(self):
        super().__init__('tracker_node')

        self.tracker_type = self.declare_parameter('tracker_type', 'bytetrack').get_parameter_value().string_value

        self.bridge = CvBridge()
        self.tracker = None

        if self.tracker_type == 'bytetrack':
            track_thresh = self.declare_parameter('track_thresh', 0.5).get_parameter_value().double_value
            match_thresh = self.declare_parameter('match_thresh', 0.8).get_parameter_value().double_value
            track_buffer = self.declare_parameter('track_buffer', 30).get_parameter_value().integer_value
            
            args = TrackerArgs(track_thresh=track_thresh, match_thresh=match_thresh, track_buffer=track_buffer)
            self.tracker = BYTETracker(args)
            self.get_logger().info("BYTETrack tracker initialized.")

        elif self.tracker_type == 'botsort':
            track_high_thresh = self.declare_parameter('track_high_thresh', 0.6).get_parameter_value().double_value
            new_track_thresh = self.declare_parameter('new_track_thresh', 0.7).get_parameter_value().double_value
            track_buffer = self.declare_parameter('track_buffer', 30).get_parameter_value().integer_value
            with_reid = self.declare_parameter('with_reid', False).get_parameter_value().bool_value
            fast_reid_config = self.declare_parameter('fast_reid_config', '').get_parameter_value().string_value
            fast_reid_weights = self.declare_parameter('fast_reid_weights', '').get_parameter_value().string_value
            
            args = TrackerArgs(track_high_thresh=track_high_thresh, new_track_thresh=new_track_thresh, 
                               track_buffer=track_buffer, with_reid=with_reid,
                               fast_reid_config=fast_reid_config, fast_reid_weights=fast_reid_weights)
            self.tracker = BoTSORT(args, frame_rate=30)
            self.get_logger().info(f"BoT-SORT tracker initialized. Re-ID enabled: {with_reid}")
        else:
            self.get_logger().error(f"Invalid tracker type: {self.tracker_type}")
            return

        # Publisher & Subscriber 설정
        self.tracked_image_pub = self.create_publisher(Image, '/tracker/tracked_image', 10)
        self.image_sub = message_filters.Subscriber(self, Image, '/rgb') # 사용하는 이미지 토픽 이름
        self.bbox_sub = message_filters.Subscriber(self, BoundingBoxes, '/yolo/detections')
        
        self.time_synchronizer = message_filters.TimeSynchronizer([self.image_sub, self.bbox_sub], 10)
        self.time_synchronizer.registerCallback(self.sync_callback)
        
        self.get_logger().info(f"Tracker Node is running with '{self.tracker_type}' tracker.")

    def sync_callback(self, image_msg: Image, bboxes_msg: BoundingBoxes):
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')
        
        # detector_node의 model.names와 순서를 맞추거나, 필요한 클래스만 정의
        CLASS_NAME_TO_ID = {
            'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3, 'bus': 5, 'truck': 7
        }
        
        detections = []
        for bbox in bboxes_msg.bounding_boxes:
            # 문자열 class_id를 숫자 ID로 변환
            class_id = CLASS_NAME_TO_ID.get(bbox.class_id, -1)

            # 맵에 없는 클래스는 추적에서 제외
            if class_id == -1:
                continue

            det = [
                bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax,
                bbox.confidence,
                float(class_id) # 변환된 숫자 ID를 float으로 전달
            ]
            detections.append(det)
        
        detections = np.array(detections) if len(detections) > 0 else np.empty((0, 6))

        online_targets = []
        if self.tracker_type == 'bytetrack':
            # BYTETrack은 class_id를 사용하지 않으므로 5개 열만 전달
            online_targets = self.tracker.update(detections[:, :5], 
                                                 [cv_image.shape[0], cv_image.shape[1]], 
                                                 (cv_image.shape[0], cv_image.shape[1]))
        elif self.tracker_type == 'botsort':
            # BoT-SORT는 원본 이미지 프레임 전체를 필요로 함
            online_targets = self.tracker.update(detections, cv_image)

        # 결과 시각화
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            tlbr = (int(tlwh[0]), int(tlwh[1]), int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3]))
            cv2.rectangle(cv_image, tlbr[:2], tlbr[2:], (0, 255, 0), 2)
            cv2.putText(cv_image, f'ID: {tid}', (tlbr[0], tlbr[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        tracked_image_msg = self.bridge.cv2_to_imgmsg(cv_image, 'bgr8')
        tracked_image_msg.header = image_msg.header
        self.tracked_image_pub.publish(tracked_image_msg)

def main(args=None):
    rclpy.init(args=args)
    node = TrackerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()