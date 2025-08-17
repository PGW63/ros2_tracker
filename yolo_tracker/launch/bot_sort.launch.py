from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():

    home_dir = os.path.expanduser('~')
    fast_reid_config_path = os.path.join(home_dir, 'ros_ws', 'src','ros2_tracker','yolo_tracker','fast_reid', 'configs', 'Market1501', 'sbs_S50.yml')
    fast_reid_weights_path = os.path.join(home_dir, 'ros_ws', 'src', 'ros2_tracker','yolo_tracker', 'models', 'mot20_sbs_S50.pth')

    return LaunchDescription([
        Node(
            package='yolo_tracker',
            executable='detector_node',
            name='detector_node',
            output='screen',
            parameters=[
                {'yolo_model': 'yolov8s.pt'}
            ]
        ),

        Node(
            package='yolo_tracker',
            executable='tracker_node',
            name='tracker_node',
            output='screen',
            parameters=[
                {'tracker_type': 'botsort'},

                {'with_reid': True},
                {'fast_reid_config': fast_reid_config_path},
                {'fast_reid_weights': fast_reid_weights_path},

                {'track_high_thresh': 0.6},
                {'new_track_thresh': 0.7},
            ]
        ),
    ])