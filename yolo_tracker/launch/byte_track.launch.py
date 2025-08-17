from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 1. YOLOv8 탐지 노드 실행
        Node(
            package='yolo_tracker',
            executable='detector_node',
            name='detector_node',
            output='screen',
            parameters=[
                # 사용할 YOLO 모델 파일 지정
                {'yolo_model': 'yolov8n.pt'} 
            ]
        ),

        # 2. 추적 노드를 BYTETrack 모드로 실행
        Node(
            package='yolo_tracker',
            executable='tracker_node',
            name='tracker_node',
            output='screen',
            parameters=[
                # 사용할 트래커 타입을 'bytetrack'으로 명시
                {'tracker_type': 'bytetrack'},

                # BYTETrack에 맞는 파라미터 설정
                {'track_thresh': 0.5},
                {'match_thresh': 0.8},
                {'track_buffer': 30},
            ]
        ),
    ])