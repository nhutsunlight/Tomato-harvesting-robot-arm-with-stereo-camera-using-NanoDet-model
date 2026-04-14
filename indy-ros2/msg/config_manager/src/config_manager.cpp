#include <rclcpp/rclcpp.hpp>
#include <yaml-cpp/yaml.h>
#include <filesystem>

#include "config_manager/msg/system_config.hpp"

using std::placeholders::_1;

class ConfigManager : public rclcpp::Node
{
public:
    ConfigManager()
    : Node("config_manager")
    {

        std::filesystem::path base_path = std::filesystem::current_path(); // sẽ là đường dẫn từ nơi bạn chạy `ros2 run`

        setup_path_ = base_path.string() + "/src/indy-ros2/msg/config_manager/config/setup.yaml";
        stereo_path_ = base_path.string() + "/src/indy-ros2/msg/config_manager/config/stereo_config.yaml";

        publisher_ = this->create_publisher<config_manager::msg::SystemConfig>(
            "/system_config",
            rclcpp::QoS(1).transient_local().reliable()
        );

        last_setup_time_ = std::filesystem::last_write_time(setup_path_);
        last_stereo_time_ = std::filesystem::last_write_time(stereo_path_);

        loadAndPublish();

        timer_ = this->create_wall_timer(
            std::chrono::seconds(2),
            std::bind(&ConfigManager::checkFiles, this)
        );
    }

private:
    void checkFiles()
    {
        auto new_setup_time = std::filesystem::last_write_time(setup_path_);
        auto new_stereo_time = std::filesystem::last_write_time(stereo_path_);

        if (new_setup_time != last_setup_time_ ||
            new_stereo_time != last_stereo_time_)
        {
            RCLCPP_INFO(this->get_logger(), "YAML changed → reloading...");
            last_setup_time_ = new_setup_time;
            last_stereo_time_ = new_stereo_time;

            loadAndPublish();
        }
    }

    void loadAndPublish()
    {
        auto msg = config_manager::msg::SystemConfig();

        // ===== setup.yaml =====
        YAML::Node setup = YAML::LoadFile(setup_path_)["setup"];

        msg.home_pose = setup["HomePose"].as<std::vector<double>>();
        msg.drop_pose = setup["DorpPose"].as<std::vector<double>>();
        msg.offset_distance = setup["OffSetDistance"].as<double>();
        msg.y_offset_distance = setup["YOffSetDistance"].as<double>();
        msg.offset_angle = setup["OffSetAngle"].as<double>();
        msg.fx_offset = setup["FxOffset"].as<double>();
        msg.object_offset = setup["ObjectOffset"].as<double>();
        msg.multi_collect_mode = setup["Multi_collect_mode"].as<bool>();

        // ===== stereo_config.yaml =====
        YAML::Node stereo = YAML::LoadFile(stereo_path_);

        msg.stereo_method = stereo["stereo_method"].as<std::string>();

        auto sgbm = stereo["stereo_sgbm"];
        msg.sgbm_min_disparity = sgbm["minDisparity"].as<int>();
        msg.sgbm_num_disparities = sgbm["numDisparities"].as<int>();
        msg.sgbm_block_size = sgbm["blockSize"].as<int>();
        msg.sgbm_p1 = sgbm["P1"].as<int>();
        msg.sgbm_p2 = sgbm["P2"].as<int>();
        msg.sgbm_disp12maxdiff = sgbm["disp12MaxDiff"].as<int>();
        msg.sgbm_uniquenessratio = sgbm["uniquenessRatio"].as<int>();
        msg.sgbm_specklewindowsize = sgbm["speckleWindowSize"].as<int>();
        msg.sgbm_specklerange = sgbm["speckleRange"].as<int>();
        msg.sgbm_prefiltercap = sgbm["preFilterCap"].as<int>();
        msg.sgbm_mode = sgbm["mode"].as<int>();

        auto bm = stereo["stereo_bm"];
        msg.bm_num_disparities = bm["numDisparities"].as<int>();
        msg.bm_block_size = bm["blockSize"].as<int>();
        msg.bm_prefiltertype = bm["preFilterType"].as<int>();
        msg.bm_prefiltersize = bm["preFilterSize"].as<int>();
        msg.bm_prefiltercap = bm["preFilterCap"].as<int>();
        msg.bm_texturethreshold = bm["textureThreshold"].as<int>();
        msg.bm_uniquenessratio = bm["uniquenessRatio"].as<int>();
        msg.bm_specklewindowsize = bm["speckleWindowSize"].as<int>();
        msg.bm_specklerange = bm["speckleRange"].as<int>();
        msg.bm_disp12maxdiff = bm["disp12MaxDiff"].as<int>();

        publisher_->publish(msg);

        RCLCPP_INFO(this->get_logger(), "Published config!");
    }

    std::string setup_path_;
    std::string stereo_path_;

    std::filesystem::file_time_type last_setup_time_;
    std::filesystem::file_time_type last_stereo_time_;

    rclcpp::Publisher<config_manager::msg::SystemConfig>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ConfigManager>());
    rclcpp::shutdown();
    return 0;
}