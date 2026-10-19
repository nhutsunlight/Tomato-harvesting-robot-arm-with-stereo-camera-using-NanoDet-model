from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='stereo_image_proc',
            executable='disparity_node',
            name='disparity_node',
            namespace='stereo',
            remappings=[
                ('left/image_rect', '/stereo/left/image_raw_calib'),
                ('left/camera_info', '/stereo/left/camera_info_calib'),
                ('right/image_rect', '/stereo/right/image_raw_calib'),
                ('right/camera_info', '/stereo/right/camera_info_calib'),
            ],
            parameters=[
                {'approximate_sync': True},
                {'disparity_range': 32},
            ],
            arguments=['--ros-args', '--log-level', 'DEBUG']
        ),
    ])