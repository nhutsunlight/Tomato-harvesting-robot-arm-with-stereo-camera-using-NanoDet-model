#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <stereo_msgs/msg/disparity_image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.hpp>
#include <opencv2/opencv.hpp>

class DepthPublisherNode : public rclcpp::Node {
public:
    DepthPublisherNode() : Node("depth_publisher_node") {
        disparity_sub_ = this->create_subscription<stereo_msgs::msg::DisparityImage>(
            "/stereo/disparity", 10, std::bind(&DepthPublisherNode::disparityCallback, this, std::placeholders::_1));

        left_imsg_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/stereo/left/image_raw_det", 10, std::bind(&DepthPublisherNode::leftImageCallback, this, std::placeholders::_1));

        depth_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/stereo/depth/image_raw", 10);
    }

private:
    rclcpp::Subscription<stereo_msgs::msg::DisparityImage>::SharedPtr disparity_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr left_imsg_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_pub_;

    void leftImageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        // Handle left image callback
        auto img = cv_bridge::toCvCopy(msg, "bgr8")->image;
        cv::imwrite("color_left_img.png", img);
    }

    void disparityCallback(const stereo_msgs::msg::DisparityImage::SharedPtr msg) {
        // Convert sensor_msgs::Image -> cv::Mat
        cv::Mat disparity(msg->image.height, msg->image.width, CV_32F,
                          reinterpret_cast<float *>(&msg->image.data[0]), msg->image.step);

    double min_val, max_val;
    cv::minMaxLoc(disparity, &min_val, &max_val);

    if (max_val == 0) {
        RCLCPP_WARN(this->get_logger(), "Disparity image is all zero!");
        return;
    }

    // ✅ Lưu float32 thô
    static int frame_count = 0;
    std::string filename = "disparity_raw_" + std::to_string(frame_count) + ".tiff";
    cv::imwrite(filename, disparity);  // tiff hỗ trợ float32 native

    // ✅ Log dùng min/max từ cv::Mat, không phải từ data bytes
    RCLCPP_INFO(this->get_logger(), "Saved: %s (min=%.2f max=%.2f)",
                filename.c_str(), min_val, max_val);


        // Kiểm tra giá trị disparity
        //double min_val, max_val;
        cv::minMaxLoc(disparity, &min_val, &max_val);

        if (max_val == 0) {
            RCLCPP_WARN(this->get_logger(), "Disparity image is all zero!");
            return;
        }

        // Chuẩn hóa disparity về 8-bit để dễ hiển thị
        cv::Mat disparity_vis;
        disparity.convertTo(disparity_vis, CV_8U, 255.0 / max_val);

        // Áp dụng colormap để nhìn dễ hơn
        cv::Mat disparity_color;
        cv::applyColorMap(disparity_vis, disparity_color, cv::COLORMAP_JET);

        // Cắt ảnh theo valid_window trong disparity message
        int x = msg->valid_window.x_offset;
        int y = msg->valid_window.y_offset;
        int w = msg->valid_window.width;
        int h = msg->valid_window.height;

        if (w > 0 && h > 0) {
            disparity_color = disparity_color(cv::Rect(x, y, w, h)).clone();
        }

        // Convert cv::Mat -> sensor_msgs::Image với encoding BGR8
        auto disparity_msg = cv_bridge::CvImage(msg->image.header, sensor_msgs::image_encodings::BGR8, disparity_color).toImageMsg();

        // Publish ảnh disparity đã chỉnh màu
        depth_pub_->publish(*disparity_msg);

        cv::imwrite("disparity_debug.png", disparity_color);
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DepthPublisherNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
