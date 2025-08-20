// dbscan.hpp – Header‑only C++17 implementation of the DBSCAN clustering algorithm for PCL point clouds
// MIT License
// Author: OpenAI ChatGPT (July 2025)

#pragma once

#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <vector>
#include <queue>
#include <unordered_set>


template <typename PointT>
class DBSCAN {
public:
    using Cloud      = pcl::PointCloud<PointT>;
    using CloudPtr   = typename Cloud::Ptr;
    using Indices    = std::vector<int>;
    using Clusters   = std::vector<Indices>;

    DBSCAN(double eps, int minPts)
        : eps_(eps), minPts_(minPts) {}

    /**
     * @brief Run DBSCAN on a point cloud.
     * @param cloud  Input cloud (KD‑tree built internally).
     * @return Vector of clusters (each cluster is a vector of indices).
     */
    Clusters operator()(const CloudPtr &cloud) {
        const int n = static_cast<int>(cloud->size());
        if (n == 0) return {};

        pcl::search::KdTree<PointT> tree;
        tree.setInputCloud(cloud);

        std::vector<bool> visited(n, false);
        std::vector<int>  labels (n, UNCLASSIFIED);

        Clusters clusters;
        Indices  neighborIndices;
        std::vector<float> neighborDists;

        for (int i = 0; i < n; ++i) {
            if (visited[i]) continue;
            visited[i] = true;

            neighborIndices.clear();
            tree.radiusSearch(i, eps_, neighborIndices, neighborDists);

            if (static_cast<int>(neighborIndices.size()) < minPts_) {
                labels[i] = NOISE;
                continue;
            }

            clusters.emplace_back();
            expandCluster(i, neighborIndices, tree, visited, labels, clusters.back(), static_cast<int>(clusters.size() - 1));
        }
        return clusters;
    }

private:
    // Special label values
    static constexpr int UNCLASSIFIED = -1;
    static constexpr int NOISE        = -2;

    double eps_;
    int    minPts_;

    void expandCluster(int           seedIndex,
                       Indices      &seedNeighbors,
                       pcl::search::KdTree<PointT> &tree,
                       std::vector<bool> &visited,
                       std::vector<int>  &labels,
                       Indices           &cluster,
                       int                clusterLabel) {
        labels[seedIndex] = clusterLabel;
        cluster.push_back(seedIndex);

        std::queue<int> Q;
        for (int idx : seedNeighbors) Q.push(idx);

        Indices neighborIndices;
        std::vector<float> neighborDists;

        while (!Q.empty()) {
            int curr = Q.front(); Q.pop();
            if (!visited[curr]) {
                visited[curr] = true;
                tree.radiusSearch(curr, eps_, neighborIndices, neighborDists);
                if (static_cast<int>(neighborIndices.size()) >= minPts_) {
                    for (int idx : neighborIndices) Q.push(idx);
                }
            }
            if (labels[curr] == UNCLASSIFIED || labels[curr] == NOISE) {
                labels[curr] = clusterLabel;
                cluster.push_back(curr);
            }
        }
    }
};
