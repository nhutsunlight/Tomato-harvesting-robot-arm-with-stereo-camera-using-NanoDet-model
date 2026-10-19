from setuptools import find_packages, setup

package_name = 'hmi_launcher'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            'launcher_gui = hmi_launcher.launcher_gui_3:main',
            'launcher_server = hmi_launcher.hmi_server:main'
        ],
    },
)
