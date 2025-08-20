from glob import glob
import os
from setuptools import find_packages, setup

package_name = 'tl_perception_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # ── ROS 패키지 인덱스 & package.xml ──
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # ── launch 디렉터리 통째로 설치 ──
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],   # + opencv-python, ultralytics 등 pip 로 이미 설치되어 있다면 OK
    zip_safe=True,
    maintainer='manggong',
    maintainer_email='manggong@todo.todo',
    description='traffic-light perception package',
    license='Apache-2.0',
    tests_require=['pytest'],

    # ── 실행 스크립트 등록 ──
    entry_points={
        'console_scripts': [
            'tl_detector_node = tl_perception_pkg.tl_detector_node:main',
            'test_node = tl_perception_pkg.test_node:main',
        ],
    },
)
