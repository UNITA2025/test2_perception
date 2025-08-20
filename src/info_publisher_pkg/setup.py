from setuptools import find_packages, setup
from glob import glob
import os


package_name = 'info_publisher_pkg'

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
    maintainer='leeseojin',
    maintainer_email='seojin1106@inu.ac.kr',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cone_info_node = info_publisher_pkg.cone_info_node:main',
            'drum_info_node = info_publisher_pkg.drum_info_node:main',
        ],
    },
)
