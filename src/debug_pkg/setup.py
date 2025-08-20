from setuptools import find_packages, setup

package_name = 'debug_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Launch files (include both primary and debug launch scripts)
        ('share/' + package_name + '/launch', [
            'launch/multi_camera_yolov8_debug.launch.py'
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hhk-laptop',
    maintainer_email='whaihong@g.skku.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'path_visualizer_node = debug_pkg.path_visualizer_node:main',
            'yolov8_visualizer_node = debug_pkg.yolov8_visualizer_node:main',
        ],
    },
)
