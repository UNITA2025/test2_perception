#include <memory>
#include <string>
#include <fstream>
#include <yaml-cpp/yaml.h>

#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/static_transform_broadcaster.h"
#include "geometry_msgs/msg/transform_stamped.hpp"

geometry_msgs::msg::TransformStamped loadTransformFromYaml(const std::string& file_path) {
    YAML::Node config = YAML::LoadFile(file_path);
    YAML::Node params = config["lidar_camera_tf"]["ros__parameters"];

    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp = rclcpp::Clock().now();
    tf.header.frame_id = params["parent_frame"].as<std::string>();
    tf.child_frame_id = params["child_frame"].as<std::string>();

    auto t = config["lidar_camera_tf"]["ros__parameters"]["translation"];
    auto r = config["lidar_camera_tf"]["ros__parameters"]["rotation"];

    tf.header.frame_id       = config["lidar_camera_tf"]["ros__parameters"]["parent_frame"].as<std::string>();
    tf.child_frame_id        = config["lidar_camera_tf"]["ros__parameters"]["child_frame"].as<std::string>();

    // ── ① translation 그대로 ──────────────────────────────
    tf.transform.translation.x = t[0].as<double>();   // x
    tf.transform.translation.y = t[1].as<double>();   // y
    tf.transform.translation.z = t[2].as<double>();   // z

    // ── ② rotation 그대로 (부호 뒤집지 않기) ────────────────
    tf.transform.rotation.x = r[0].as<double>();      // qx
    tf.transform.rotation.y = r[1].as<double>();      // qy
    tf.transform.rotation.z = r[2].as<double>();      // qz  ← 부호 수정 안 함
    tf.transform.rotation.w = r[3].as<double>();      // qw  ← 부호 수정 안 함


    return tf;
}

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("lidar_to_camera_tf_broadcaster");
    tf2_ros::StaticTransformBroadcaster broadcaster(node);

    std::string base_path = node->declare_parameter("config_path", "/home/mangggong/ros2_ws/src/sensor_initialize/config/");

    std::vector<std::string> file_names = {
        "lidar_to_cam1_front_up_tf.yaml",
        "lidar_to_cam2_front_down_tf.yaml",
        "lidar_to_cam3_left_tf.yaml",
        "lidar_to_cam4_right_tf.yaml"
    };

    for (const auto& file_name : file_names) {
        std::string file_path = base_path + file_name;
        RCLCPP_INFO(node->get_logger(), "Loading: %s", file_path.c_str());

        try {
            auto tf = loadTransformFromYaml(file_path);
            broadcaster.sendTransform(tf);
        } catch (const YAML::Exception& e) {
            RCLCPP_ERROR(node->get_logger(), "Failed to parse %s: %s", file_path.c_str(), e.what());
        }
    }

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}