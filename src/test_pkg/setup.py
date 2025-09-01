from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'test_pkg'

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
    maintainer='unita',
    maintainer_email='junssong@student.42seoul.kr',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'test = test_pkg.test:main',
            'test_visualizer = test_pkg.test_visualizer:main',
            'test_planner = test_pkg.test_planner:main',
            'test_follower = test_pkg.test_follower:main',
        ],
    },
)

