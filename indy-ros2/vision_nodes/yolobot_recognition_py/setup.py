import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'yolobot_recognition_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
            (os.path.join('share', 'yolobot_recognition_py', 'models'),
        glob('model/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nhut',
    maintainer_email='nhut@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'recognition_node = yolobot_recognition_py.yolo_with_ros2:main',
        ],
    },
)
