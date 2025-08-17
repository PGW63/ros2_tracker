from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'yolo_tracker'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch*.*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gw',
    maintainer_email='gw@todo.todo',
    description='YOLOv8 and BYTETrack with ROS2',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detector_node = yolo_tracker.nodes.detector_node:main',
            'tracker_node = yolo_tracker.nodes.tracker_node:main',
            'video_publisher_loop = yolo_tracker.nodes.video_pub_node:main',
        ],
    },
)