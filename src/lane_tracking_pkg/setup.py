from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'lane_tracking_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='seonu',
    maintainer_email='rkscjq21@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'image_publisher_node = lane_tracking_pkg.image_publisher_node:main',
            'lane_info_extractor_node = lane_tracking_pkg.lane_info_extractor_node:main',
            'path_visualizer_node = lane_tracking_pkg.path_visualizer_node:main',
            'yolov8_node = lane_tracking_pkg.yolov8_node:main',
            'yolov8_visualizer_node = lane_tracking_pkg.yolov8_visualizer_node:main'
        ],
    }
)
