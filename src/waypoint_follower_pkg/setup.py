from setuptools import setup
from glob import glob
import os

package_name = 'waypoint_follower_pkg'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        # ✅ ament 패키지 인덱스 마커 설치 (필수)
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        # ✅ package.xml 설치
        ('share/' + package_name, ['package.xml']),
        # (있으면) launch 파일도 같이 설치
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gyun',
    maintainer_email='gyun@todo.todo',
    description='Waypoint follower package',
    license='MIT',  # TODO에서 실제로 바꾸기
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'waypoint_extractor = waypoint_follower_pkg.waypoint_extractor:main',
            'command_publisher = waypoint_follower_pkg.command_publisher:main',
            'sim_erp_bridge   = waypoint_follower_pkg.sim_erp_bridge:main',
            'cone_info_debug_pub = waypoint_follower_pkg.cone_info_debug_pub:main',
        ],
    },
)
