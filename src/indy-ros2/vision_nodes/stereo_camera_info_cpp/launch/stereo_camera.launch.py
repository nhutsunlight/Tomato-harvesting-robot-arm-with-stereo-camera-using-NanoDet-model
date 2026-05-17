from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="stereo_camera_info_cpp",
            executable="stereo_camera_info_node",
            name="stereo_camera_info_publisher",
            parameters=[{
                "left_camera_info_url": "config/left.yaml",
                "right_camera_info_url": "config/right.yaml"
            }],
            remappings=[
                ("/left/camera_info", "/left/camera_info_final"),
                ("/right/camera_info", "/right/camera_info_final")
            ]
        )
    ])