#==================================================#
# 패키지(Package)
# - 이름: dead_reckoning_pkg
# - 버전: 0.0.0
#
# 실행 엔트리포인트(Console scripts)
# - pointcloud_filter_node             → dead_reckoning_pkg/pointcloud_filter_node.py:main
# - pointcloud_filter_visualizer_node  → dead_reckoning_pkg/pointcloud_filter_visualizer_node.py:main
# - mapping_node                       → dead_reckoning_pkg/mapping_node.py:main
# - lane_planner_node                  → dead_reckoning_pkg/lane_planner_node.py:main
# - path_follower_node                 → dead_reckoning_pkg/path_follower_node.py:main
#
# - test_follower                      → dead_reckoning_pkg/test_follower.py:main
#
# TODO :
# 최종 수정일: 2025.09.19
# 편집자: 이기현, 정선우
#==================================================#

from setuptools import find_packages, setup
from glob import glob
import os
package_name = 'dead_reckoning_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['dead_reckoning']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='unita',
    maintainer_email='junssong@student.42seoul.kr',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pointcloud_filter_node = dead_reckoning_pkg.pointcloud_filter_node:main',
            'pointcloud_filter_visualizer_node = dead_reckoning_pkg.pointcloud_filter_visualizer_node:main',     
            'mapping_node = dead_reckoning_pkg.mapping_node:main',
            'lane_planner_node = dead_reckoning_pkg.lane_planner_node:main',
            'path_follower_node = dead_reckoning_pkg.path_follower_node:main',

            'test_follower = dead_reckoning_pkg.test_follower:main',
        ],
    },
)

