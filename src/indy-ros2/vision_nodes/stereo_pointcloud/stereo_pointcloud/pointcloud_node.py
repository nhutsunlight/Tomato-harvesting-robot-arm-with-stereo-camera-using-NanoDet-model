import rclpy
import cv2
import numpy as np
import struct
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField, CameraInfo
from stereo_msgs.msg import DisparityImage
from cv_bridge import CvBridge


class StereoPointCloud(Node):
    def __init__(self):
        super().__init__('stereo_pointcloud')

        # ROS 2 Bridge for image conversion
        self.bridge = CvBridge()

        # Camera parameters
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.baseline = None

        # Subscribe to CameraInfo topics
        self.create_subscription(CameraInfo, '/stereo/left/camera_info_calib', self.left_camera_info_callback, 10)
        self.create_subscription(CameraInfo, '/stereo/right/camera_info_calib', self.right_camera_info_callback, 10)

        # Subscribe to disparity image
        self.create_subscription(DisparityImage, '/stereo/disparity', self.process_points2, 1)  # Reduce queue size

        # Publisher for PointCloud2
        self.point_cloud_pub = self.create_publisher(PointCloud2, '/points2', 10)

        self.get_logger().info("StereoPointCloud node started!")

    def left_camera_info_callback(self, msg):
        """Nhận thông tin từ /stereo/left/camera_info"""
        self.fx = msg.k[0]  # Tiêu cự theo trục X
        self.fy = msg.k[4]  # Tiêu cự theo trục Y
        self.cx = msg.k[2]  # Principal point X
        self.cy = msg.k[5]  # Principal point Y

    def right_camera_info_callback(self, msg):
        """Nhận thông tin từ /stereo/right/camera_info để tính baseline"""
        self.baseline = abs(msg.p[3]) / msg.p[0]  # Baseline = -Tx / fx

    def process_points2(self, disparity_msg):
        """Convert disparity map to PointCloud2 and publish."""
        if self.fx is None or self.baseline is None:
            self.get_logger().warn("Waiting for CameraInfo messages!")
            return

        # Convert disparity image
        disparity = self.bridge.imgmsg_to_cv2(disparity_msg.image, desired_encoding="32FC1")

        # Tạo điểm point cloud từ disparity
        point_cloud_msg = self.generate_point_cloud(disparity)

        # Publish point cloud
        if point_cloud_msg is not None:
            self.point_cloud_pub.publish(point_cloud_msg)
            self.get_logger().info("Published PointCloud2")

    def generate_point_cloud(self, disparity):
        """Convert disparity image to 3D point cloud."""
        height, width = disparity.shape

        # 1. **Tạo Meshgrid tọa độ ảnh**
        v, u = np.meshgrid(np.arange(width), np.arange(height))

        # 2. **Tính toán depth (Z) từ disparity**
        valid_disp = disparity > 0
        Z = np.where(valid_disp, (self.fx * self.baseline) / disparity, np.nan)

        # 3. **Giới hạn khoảng cách Z để tránh lỗi đo**
        max_z = 3.0  # Giới hạn điểm xa nhất là 3m
        Z[Z > max_z] = np.nan  # Lọc các điểm xa

        # 4. **Tính tọa độ X, Y**
        X = (u - self.cx) * Z / self.fx
        Y = (v - self.cy) * Z / self.fy

        # 5. **Làm phẳng dữ liệu**
        X, Y, Z = X.flatten(), Y.flatten(), Z.flatten()

        # 6. **Lọc các điểm hợp lệ**
        mask = ~np.isnan(Z)
        X, Y, Z = X[mask], Y[mask], Z[mask]

        # 7. **Giảm số lượng điểm để tránh crash RViz**
        step = 16  # Mỗi 5 điểm lấy 1 điểm
        X, Y, Z = X[::step], Y[::step], Z[::step]

        # 8. **Gán màu mặc định**
        rgb = np.full_like(X, fill_value=struct.unpack('I', struct.pack('BBBB', 255, 255, 255, 0))[0])

        # 9. **Tạo PointCloud2**
        cloud_data = np.stack((Z,-Y,-X, rgb), axis=-1).astype(np.float32)
        point_cloud_msg = self.create_pointcloud2_msg(cloud_data)

        return point_cloud_msg

    def create_pointcloud2_msg(self, cloud_data):
        """Tạo message PointCloud2 từ NumPy array"""
        point_cloud_msg = PointCloud2()
        point_cloud_msg.header.frame_id = "stereo_camera_base"
        point_cloud_msg.height = 1
        point_cloud_msg.width = cloud_data.shape[0]
        point_cloud_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]
        point_cloud_msg.is_bigendian = False
        point_cloud_msg.point_step = 16
        point_cloud_msg.row_step = point_cloud_msg.point_step * cloud_data.shape[0]
        point_cloud_msg.data = cloud_data.tobytes()
        point_cloud_msg.is_dense = False

        return point_cloud_msg


def main(args=None):
    rclpy.init(args=args)
    node = StereoPointCloud()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
