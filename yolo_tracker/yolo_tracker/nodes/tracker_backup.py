# yolo_tracker/yolo_tracker/nodes/tracker_node.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2

# 동기화를 위한 message_filters
import message_filters

# 커스텀 메시지 및 트래커 클래스 임포트
from vision_interfaces.msg import BoundingBox, BoundingBoxes
from ..tracker.byte_track.byte_tracker import BYTETracker

# BYTETracker의 'args' 객체를 대체할 간단한 클래스
class TrackerArgs:
    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8, mot20=False):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.mot20 = mot20

class ByteTrackerNode(Node):
    def __init__(self):
        super().__init__('byte_tracker_node')
        
        # 트래커 초기화 (ROS 파라미터로 설정 가능하게)
        self.declare_parameter('track_thresh', 0.5)
        track_thresh = self.get_parameter('track_thresh').get_parameter_value().double_value
        
        args = TrackerArgs(track_thresh=track_thresh)
        self.tracker = BYTETracker(args)
        self.bridge = CvBridge()

        # 시각화된 이미지를 발행할 Publisher
        self.tracked_image_pub = self.create_publisher(Image, '/tracker/tracked_image', 10)

        # Subscriber (이미지와 바운딩 박스를 동기화하여 수신)
        self.image_sub = message_filters.Subscriber(self, Image, '/rgb')
        self.bbox_sub = message_filters.Subscriber(self, BoundingBoxes, '/yolo/detections')
        
        self.time_synchronizer = message_filters.TimeSynchronizer(
            [self.image_sub, self.bbox_sub], 10)
        self.time_synchronizer.registerCallback(self.sync_callback)
        
        self.get_logger().info("BYTETracker Node is running.")

    def sync_callback(self, image_msg: Image, bboxes_msg: BoundingBoxes):
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')
        
        # 수신한 바운딩 박스 메시지를 BYTETracker 입력 형식으로 변환
        detections = []
        for bbox in bboxes_msg.bounding_boxes:
            # 형식: [x1, y1, x2, y2, score]
            det = [
                bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax, bbox.confidence
            ]
            detections.append(det)
        
        detections = np.array(detections) if len(detections) > 0 else np.empty((0, 5))
        
        # 트래커 업데이트
        online_targets = self.tracker.update(detections, [cv_image.shape[0], cv_image.shape[1]], (cv_image.shape[0], cv_image.shape[1]))
        
        # 결과 시각화
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            # 바운딩 박스 그리기 (tlwh -> tlbr)
            tlbr = (int(tlwh[0]), int(tlwh[1]), int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3]))
            cv2.rectangle(cv_image, tlbr[:2], tlbr[2:], (0, 255, 0), 2)
            # 트랙 ID 표시
            cv2.putText(cv_image, f'ID: {tid}', (tlbr[0], tlbr[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 시각화된 이미지를 ROS 메시지로 변환하여 발행
        tracked_image_msg = self.bridge.cv2_to_imgmsg(cv_image, 'bgr8')
        tracked_image_msg.header = image_msg.header
        self.tracked_image_pub.publish(tracked_image_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ByteTrackerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()