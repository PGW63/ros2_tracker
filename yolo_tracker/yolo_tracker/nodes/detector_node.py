# yolo_tracker/yolo_tracker/nodes/detector_node.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO

from vision_interfaces.msg import BoundingBox, BoundingBoxes

class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detector_node')
        
        self.declare_parameter('yolo_model', 'yolov8s.pt')
        model_path = self.get_parameter('yolo_model').get_parameter_value().string_value
        
        self.model = YOLO(model_path)
        self.bridge = CvBridge()
        
        self.image_sub = self.create_subscription(
            Image,
            '/rgb', 
            self.image_callback,
            10)
        self.bbox_pub = self.create_publisher(BoundingBoxes, '/yolo/detections', 10)
        
        self.get_logger().info(f"YOLOv8 Detector Node is running with model: {model_path}")

    def image_callback(self, msg: Image):
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        
        results = self.model(cv_image, verbose=False)
        
        bboxes_msg = BoundingBoxes()
        bboxes_msg.header = msg.header  
        
        for result in results:
            for box in result.boxes:
                bbox_msg = BoundingBox()
                bbox_msg.class_id = self.model.names[int(box.cls)]
                bbox_msg.confidence = float(box.conf)
                coords = box.xyxy[0].cpu().numpy()
                bbox_msg.xmin = float(coords[0])
                bbox_msg.ymin = float(coords[1])
                bbox_msg.xmax = float(coords[2])
                bbox_msg.ymax = float(coords[3])
                bboxes_msg.bounding_boxes.append(bbox_msg)
        
        #Publish 
        if len(bboxes_msg.bounding_boxes) > 0:
            self.bbox_pub.publish(bboxes_msg)

def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()