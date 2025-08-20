#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>

#include <pcl/filters/crop_box.h>     // ← CropBox 선언
#include <pcl/common/centroid.h>      // ← compute3DCentroid, demeanPointCloud

class LidarPreprocessor : public rclcpp::Node
{
public:
    LidarPreprocessor()
    : Node("lidar_preprocessing_node")
    {
        // Left LiDAR
        sub_left_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/velodyne_points", 10, std::bind(&LidarPreprocessor::callbackLeft, this, std::placeholders::_1));
        pub_left_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/lidar_pre_left", 10);

        // Right LiDAR
        sub_right_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/velodyne_points", 10, std::bind(&LidarPreprocessor::callbackRight, this, std::placeholders::_1));
        pub_right_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/lidar_pre_right", 10);

        // Down LiDAR
        sub_down_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/velodyne_points", 10, std::bind(&LidarPreprocessor::callbackDown, this, std::placeholders::_1));
        pub_down_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/lidar_pre_down", 10);
    }

private:
    // 공통 Voxel Grid 크기
    const float voxel_leaf_size_ = 0.05f;

    // LiDAR별 ROI 설정
    struct ROI {
        float min_x, max_x;
        float min_y, max_y;
        float min_z, max_z;
    };

    ROI roi_left_{-1.0f, 5.5f, 0.0f, 3.0f, -0.5f, 1.0f};
    ROI roi_right_{-1.0f, 5.5f, -3.0f, 0.0f, -0.5f, 1.0f};
    ROI roi_down_{-0.5f, 7.0f, -4.0f, 4.0f, -0.4f, 1.0f};

    void callbackLeft(const sensor_msgs::msg::PointCloud2::SharedPtr msg)  { process(msg, roi_left_, pub_left_); }
    void callbackRight(const sensor_msgs::msg::PointCloud2::SharedPtr msg) { process(msg, roi_right_, pub_right_); }
    void callbackDown(const sensor_msgs::msg::PointCloud2::SharedPtr msg)  { process(msg, roi_down_, pub_down_); }

    void process(const sensor_msgs::msg::PointCloud2::SharedPtr input,
                 const ROI& roi,
                 rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub)
    {
        // 0) ROS → PCL
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::fromROSMsg(*input, *cloud_in);

        // 1) ROI 먼저 (CropBox 권장)
        pcl::CropBox<pcl::PointXYZI> crop;
        crop.setMin(Eigen::Vector4f(roi.min_x, roi.min_y, roi.min_z, 1.0f));
        crop.setMax(Eigen::Vector4f(roi.max_x, roi.max_y, roi.max_z, 1.0f));
        crop.setInputCloud(cloud_in);
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_roi(new pcl::PointCloud<pcl::PointXYZI>());
        crop.filter(*cloud_roi);

        // 로컬화: 원점 근처로 옮겨 인덱스 범위 축소
        Eigen::Vector4f c; pcl::compute3DCentroid(*cloud_roi, c);
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_local(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::demeanPointCloud<pcl::PointXYZI>(*cloud_roi, c, *cloud_local);

        // 2) VoxelGrid (리프 조금 키워서 시작)
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_voxel(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::VoxelGrid<pcl::PointXYZI> voxel;
        voxel.setInputCloud(cloud_local);
        voxel.setLeafSize(0.1f, 0.1f, 0.1f); // 0.05f가 필요하면 나중에 줄여보기
        voxel.filter(*cloud_voxel);

        // 다시 원래 위치로 복원
        for (auto &p : cloud_voxel->points) {
            p.x += c[0]; p.y += c[1]; p.z += c[2];
        }

        // 3) PCL → ROS
        sensor_msgs::msg::PointCloud2 out;
        pcl::toROSMsg(*cloud_voxel, out);
        out.header = input->header;
        pub->publish(out);
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_left_, sub_right_, sub_down_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_left_, pub_right_, pub_down_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LidarPreprocessor>());
    rclcpp::shutdown();
    return 0;
}

