#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>

#include "dbscan.h"   // ★ 그대로 사용

using std::placeholders::_1;

class ClusterNode : public rclcpp::Node
{
public:
  ClusterNode()
  : Node("cluster_node")
  {
    /* ──── 1. 파라미터 선언 & 읽기 ──── */
    declare_parameter<double>("eps", 0.3);
    declare_parameter<int>("min_pts", 10);
    declare_parameter<int>("max_pts", 1000);

    eps_      = get_parameter("eps").as_double();
    min_pts_  = get_parameter("min_pts").as_int();
    max_pts_  = get_parameter("max_pts").as_int();

    /* ──── 2. 퍼블리셔 / 서브스크라이버 ──── */
    auto qos   = rclcpp::SensorDataQoS(rclcpp::KeepLast(1));   // PointCloud2 에 권장
    sub_cloud_ = create_subscription<sensor_msgs::msg::PointCloud2>(
                   "/lidar_preprocessed", qos,
                   std::bind(&ClusterNode::cloudCallback, this, _1));

    pub_markers_ = create_publisher<visualization_msgs::msg::MarkerArray>(
                     "/cluster_markers",  1);
    pub_centroids_ = create_publisher<sensor_msgs::msg::PointCloud2>(
                      "/cluster_centroids", 1);
  }

private:
  /* ──── 콜백 ──── */
  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg_in)
  {
    // 1) ROS2 → PCL 변환
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*msg_in, *cloud);

    // 2) DBSCAN
    DBSCAN<pcl::PointXYZI> dbscan(eps_, min_pts_);
    auto clusters = dbscan(cloud);

    // PCL PointIndices 변환(시각화를 위해)
    std::vector<pcl::PointIndices> cluster_indices;
    for (const auto &idx : clusters) {
      pcl::PointIndices pi;
      pi.indices.assign(idx.begin(), idx.end());
      cluster_indices.emplace_back(std::move(pi));
    }

    // 3) 결과 메시지 구성
    visualization_msgs::msg::MarkerArray markers;
    pcl::PointCloud<pcl::PointXYZ> centers;
    int id = 0;

    for (const auto &ci : cluster_indices)
    {
      pcl::PointCloud<pcl::PointXYZI>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZI>);
      for (int i : ci.indices) cluster->push_back((*cloud)[i]);

      // 중심점 계산
      Eigen::Vector4f centroid;
      pcl::compute3DCentroid(*cluster, centroid);

      // Marker 생성
      visualization_msgs::msg::Marker m;
      m.header          = msg_in->header;
      m.ns              = "dbscan_center";
      m.id              = id++;
      m.type            = visualization_msgs::msg::Marker::SPHERE;
      m.action          = visualization_msgs::msg::Marker::ADD;
      m.pose.position.x = centroid[0];
      m.pose.position.y = centroid[1];
      m.pose.position.z = centroid[2];
      m.scale.x = m.scale.y = m.scale.z = 0.2;
      m.color.r = 1.0f;  m.color.g = 0.0f;  m.color.b = 0.0f;  m.color.a = 1.0f;
      m.lifetime        = rclcpp::Duration::from_seconds(0.5);
      markers.markers.push_back(m);

      centers.push_back(pcl::PointXYZ(centroid[0], centroid[1], centroid[2]));
    }

    // 4) 퍼블리시
    pub_markers_->publish(markers);

    sensor_msgs::msg::PointCloud2 msg_centroids;
    pcl::toROSMsg(centers, msg_centroids);
    msg_centroids.header = msg_in->header;
    pub_centroids_->publish(msg_centroids);
  }

  /* ──── 멤버 변수 ──── */
  double eps_;
  int    min_pts_;
  int    max_pts_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_cloud_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr       pub_centroids_;
};

/* ──── main ──── */
int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ClusterNode>());
  rclcpp::shutdown();
  return 0;
}
