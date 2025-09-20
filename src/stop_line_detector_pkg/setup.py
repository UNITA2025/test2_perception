from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'stop_line_detector_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
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
            'stop_line_detector_node = stop_line_detector_pkg.stop_line_detector_node:main',
            'video_publisher_node = stop_line_detector_pkg.video_publisher_node:main',
        ],
    },
)
