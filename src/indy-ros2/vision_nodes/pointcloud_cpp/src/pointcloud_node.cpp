#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <stereo_msgs/msg/disparity_image.hpp>
#include <image_geometry/stereo_camera_model.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

class StereoPointCloudNode : public rclcpp::Node {
public:
    StereoPointCloudNode() : Node("stereo_pointcloud_node") {
        using std::placeholders::_1;

        // Subscribe to image, camera info, and disparity topics
        left_image_sub_ = create_subscription<sensor_msgs::msg::Image>(
            "/stereo/left/img_for_yolo", 10, std::bind(&StereoPointCloudNode::image_callback, this, _1));

        left_camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
            "/stereo/left/camera_info_calib", 10,
            std::bind(&StereoPointCloudNode::left_camera_info_callback, this, _1));

        right_camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
            "/stereo/right/camera_info_calib", 10,
            std::bind(&StereoPointCloudNode::right_camera_info_callback, this, _1));

        disparity_sub_ = create_subscription<stereo_msgs::msg::DisparityImage>(
            "/stereo/disparity", 10, std::bind(&StereoPointCloudNode::disparity_callback, this, _1));

        // Publisher for PointCloud2
        point_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("/stereo/points2", 10);
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr left_image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr left_camera_info_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr right_camera_info_sub_;
    sensor_msgs::msg::CameraInfo::SharedPtr left_camera_info_;
    sensor_msgs::msg::CameraInfo::SharedPtr right_camera_info_;
    rclcpp::Subscription<stereo_msgs::msg::DisparityImage>::SharedPtr disparity_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_pub_;

    image_geometry::StereoCameraModel model_;
    cv::Mat left_image_;

    void left_camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        left_camera_info_ = std::make_shared<sensor_msgs::msg::CameraInfo>(*msg);
        update_camera_model();
    }
    
    void right_camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        right_camera_info_ = std::make_shared<sensor_msgs::msg::CameraInfo>(*msg);
        update_camera_model();
    }
    
    void update_camera_model() {
        if (left_camera_info_ && right_camera_info_) {
            model_.fromCameraInfo(*left_camera_info_, *right_camera_info_);
            RCLCPP_INFO(this->get_logger(), "Stereo camera model updated!");
        }
    }

    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        try {
            left_image_ = cv_bridge::toCvCopy(msg, "bgr8")->image;
        } catch (cv_bridge::Exception &e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }

    void disparity_callback(const stereo_msgs::msg::DisparityImage::SharedPtr disp_msg) {
        if (left_image_.empty()) {
            //RCLCPP_WARN(this->get_logger(), "Chưa nhận được ảnh trái!");
            return;
        }

        // Convert disparity image to OpenCV format
        const sensor_msgs::msg::Image &dimage = disp_msg->image;
        float *data = reinterpret_cast<float *>(const_cast<uint8_t *>(&dimage.data[0]));
        cv::Mat disparity(dimage.height, dimage.width, CV_32F, data, dimage.step);

        // Chuyển disparity thành 3D point cloud
        cv::Mat points_mat;
        model_.projectDisparityImageTo3d(disparity, points_mat, true);

        // Xuất ra PointCloud2
        auto point_cloud_msg = generate_point_cloud(points_mat, left_image_);
        point_cloud_pub_->publish(point_cloud_msg);
        RCLCPP_INFO(this->get_logger(), "Published PointCloud2");
    }

    sensor_msgs::msg::PointCloud2 generate_point_cloud(const cv::Mat &points, const cv::Mat &color_img) {
        sensor_msgs::msg::PointCloud2 cloud_msg;
        //cloud_msg.header.stamp = this->get_clock()->now();
        cloud_msg.header.frame_id = "stereo_camera_base";

        cloud_msg.height = points.rows;
        cloud_msg.width = points.cols;
        cloud_msg.fields.resize(4);
        cloud_msg.fields[0].name = "x";
        cloud_msg.fields[0].offset = 0;
        cloud_msg.fields[0].count = 1;
        cloud_msg.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
        cloud_msg.fields[1].name = "y";
        cloud_msg.fields[1].offset = 4;
        cloud_msg.fields[1].count = 1;
        cloud_msg.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
        cloud_msg.fields[2].name = "z";
        cloud_msg.fields[2].offset = 8;
        cloud_msg.fields[2].count = 1;
        cloud_msg.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
        cloud_msg.fields[3].name = "rgb";
        cloud_msg.fields[3].offset = 12;
        cloud_msg.fields[3].count = 1;
        cloud_msg.fields[3].datatype = sensor_msgs::msg::PointField::FLOAT32;
        // points.is_bigendian = false; ???
        cloud_msg.point_step = 16;
        cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width;
        cloud_msg.data.resize(cloud_msg.row_step * cloud_msg.height);
        cloud_msg.is_dense = false;

        sensor_msgs::PointCloud2Modifier pcd_modifier(cloud_msg);
        pcd_modifier.setPointCloud2Fields(4,
                                          "x", 1, sensor_msgs::msg::PointField::FLOAT32,
                                          "y", 1, sensor_msgs::msg::PointField::FLOAT32,
                                          "z", 1, sensor_msgs::msg::PointField::FLOAT32,
                                          "rgb", 1, sensor_msgs::msg::PointField::FLOAT32);

        sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");
        sensor_msgs::PointCloud2Iterator<uint8_t> iter_r(cloud_msg, "r");
        sensor_msgs::PointCloud2Iterator<uint8_t> iter_g(cloud_msg, "g");
        sensor_msgs::PointCloud2Iterator<uint8_t> iter_b(cloud_msg, "b");

        float bad_point = std::numeric_limits<float>::quiet_NaN();
        int step = 8;
        for (int v = 0; v < points.rows;  v += step) {
            for (int u = 0; u < points.cols;  
                u += step, ++iter_x, ++iter_y, ++iter_z, 
                ++iter_r, ++iter_g, ++iter_b
            ) {
                cv::Vec3f pt = points.at<cv::Vec3f>(v, u);
                if (pt[2] > 0.1 && pt[2] < 2.0) {  // Giới hạn phạm vi Z từ 10cm đến 3m

                    *iter_x = pt[2];  // Z -> X
                    *iter_y = -pt[0]; // -Y -> Y
                    *iter_z = -pt[1]; // -X -> Z

                    cv::Vec3b color = color_img.at<cv::Vec3b>(v, u);

                    *iter_r = color[2];
                    *iter_g = color[1];
                    *iter_b = color[0];
                } else {

                    *iter_x = *iter_y = *iter_z = bad_point;
                }
            }
        }

        return cloud_msg;
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<StereoPointCloudNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
