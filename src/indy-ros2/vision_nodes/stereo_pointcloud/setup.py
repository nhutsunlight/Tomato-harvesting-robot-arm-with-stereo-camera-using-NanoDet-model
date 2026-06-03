from setuptools import find_packages, setup

package_name = 'stereo_pointcloud'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/stereo_pointcloud/config/',
            ['config/' + 'left.yaml']),
        ('share/stereo_pointcloud/config/',
            ['config/' + 'right.yaml']),
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
            'pointcloud_node = stereo_pointcloud.pointcloud_node:main',
        ],
    },
)
