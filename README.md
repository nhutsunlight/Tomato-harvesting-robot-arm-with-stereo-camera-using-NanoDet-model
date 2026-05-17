# Tomato Harvesting Robot Arm with Stereo Camera using NanoDet

A ROS2-based robotic system for autonomous tomato harvesting using stereo vision and the NanoDet model for object detection.

## Features
- **Stereo Vision**: Dual camera setup for 3D perception
- **NanoDet Model**: Lightweight object detection for tomato identification
- **NeuroMeka Indy7**: 6-DOF collaborative robot arm
- **Gazebo Simulation**: Full simulation environment with physics
- **Web Interface**: Flask-based monitoring and control
- **QR Code Support**: Navigation and tracking capabilities

## Installation

### ROS2 Dependencies
```bash
sudo apt install ros-humble-camera-info-manager
sudo apt-get install ros-humble-stereo-image-proc
sudo apt install ros-humble-image-pipeline
sudo apt install ros-humble-ros-ign-bridge
sudo apt-get install ros-humble-ign-ros2-control
sudo apt install libompl-dev
```

### Python Dependencies
```bash
pip install opencv-python-headless
pip install numpy==1.26.4
pip install ttkbootstrap
pip install flask
pip install pyngrok
pip install qrcode[pil]
pip install customtkinter
pip install onnxruntime
pip install flask-socketio
```

### NeuroMeka Indy ROS2
Install the NeuroMeka Indy ROS2 package according to their official documentation.

## Quick Start

Launch the simulation environment:
```bash
ros2 launch indy_moveit indy_moveit_gazebo.launch.py indy_type:=indy7
```

## Tech Stack
- **Language**: Python, C++, CMake, Shell
- **Framework**: ROS2 Humble
- **Simulation**: Gazebo
- **Detection**: NanoDet
- **Robot**: NeuroMeka Indy7
