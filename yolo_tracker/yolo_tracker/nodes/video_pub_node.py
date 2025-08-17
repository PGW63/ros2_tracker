import rclpy
from rclpy.node import Node
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class VideoPublisherLoop(Node):
    """
    비디오 파일을 무한 반복하여 읽고 ROS 2 토픽으로 발행하는 노드입니다.
    """
    def __init__(self):
        super().__init__('video_publisher_loop')
        
        self.publisher_ = self.create_publisher(Image, 'rgb', 10)
        
        # 30 FPS로 이미지를 발행하기 위해 타이머를 설정합니다.
        timer_period = 1.0 / 30.0
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        # ⭐ 중요: 'path/to/your/video.mp4' 부분을 실제 영상 파일 경로로 바꿔주세요.
        video_path = '/home/gw/Videos/walking.mp4'
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            self.get_logger().error(f"비디오 파일을 열 수 없습니다: {video_path}")
            rclpy.shutdown()
        
        self.br = CvBridge()
        self.get_logger().info('비디오 무한 반복 퍼블리셔 노드가 시작되었습니다.')

    def timer_callback(self):
        """
        타이머에 의해 주기적으로 호출되어 한 프레임을 읽고 발행합니다.
        """
        # 비디오에서 한 프레임을 읽어옵니다.
        ret, frame = self.cap.read()
        
        # ⭐ 핵심 수정 사항: 영상이 끝났는지 확인하고, 끝났으면 처음으로 되돌립니다.
        if not ret:
            self.get_logger().info('비디오가 끝나서 처음부터 다시 재생합니다.')
            # 프레임 위치를 0 (맨 처음)으로 설정합니다.
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            # 되돌린 후 첫 프레임을 다시 읽어옵니다.
            ret, frame = self.cap.read()

        # 성공적으로 프레임을 읽었다면 토픽으로 발행합니다.
        if ret:
            ros_image_msg = self.br.cv2_to_imgmsg(frame, "bgr8")
            ros_image_msg.header.stamp = self.get_clock().now().to_msg()
            self.publisher_.publish(ros_image_msg)

def main(args=None):
    rclpy.init(args=args)
    video_publisher = VideoPublisherLoop()
    rclpy.spin(video_publisher)
    video_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()