//Francesco Campigotto

#include "TestsDetector.hpp"
#include "preprocessing.hpp"
#include <iostream>
#include <filesystem>
#include <queue>
#include <unordered_set>
#include <algorithm>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

vector<pair<Rect, string>> detectObjects(
    const Mat &scene,
    const string &sceneName,
    const vector<ObjectModel> &models,
    Ptr<Feature2D> &detector
) {
    // Configuration constants: Define thresholds, parameters, and limits for various processing steps
    constexpr float MATCH_RATIO_THRESHOLD = 0.85f;  // For matching descriptors
    constexpr int MIN_INLIERS = 4;  // Minimum inliers for homography validation
    constexpr double RANSAC_THRESHOLD = 5.0;  // Threshold for RANSAC homography estimation
    constexpr float HOMOGRAPHY_DET_THRESHOLD = 0.1;  // Lower limit for homography determinant
    constexpr float HOMOGRAPHY_DET_UPPER_THRESHOLD = 10.0;  // Upper limit for homography determinant
    constexpr float CLUSTER_DISTANCE_THRESHOLD = 20.0f;  // Distance threshold for clustering
    constexpr int MIN_POINTS_PER_CLUSTER = 20;  // Minimum number of points to consider a valid cluster
    constexpr float BOX_MERGE_DISTANCE = 250.0f;  // Maximum distance between bounding boxes for merging
    const int MAX_BOX_AREA = 1600;  // Maximum allowed area for valid bounding boxes
    const float DYNAMIC_MARGIN = 1.5f;  // Factor for dynamic margin based on points' standard deviation

    // Preprocess the scene for object detection
    Mat preprocessedScene = preprocessImage(scene);
    vector<pair<Rect, string>> detections;

    BFMatcher matcher(NORM_L2);  // Feature matcher using L2 norm

    for (const auto &model : models) {  // Iterate over the object models
        vector<Point2f> allUnfilteredScenePts;  // Holds all matched points from scene
        vector<Point2f> discardedPtsGlobal;  // Points that do not belong to valid clusters

        // Multiscale detection for matching object model with scene
        auto detectAtScale = [&](const Mat& sceneImg, const vector<KeyPoint>& kp, const Mat& desc, float scale = 1.0f) {
            // Function to detect objects at different scales (zoom in/out)
            const vector<KeyPoint>& sceneKP = kp;
            const Mat& sceneDesc = desc;

            auto scalePoints = [scale](vector<Point2f>& points) {
                if (scale != 1.0f) {
                    for (auto& pt : points) {
                        pt.x /= scale;
                        pt.y /= scale;
                    }
                }
            };

            // Matching between object model and scene
            for (size_t i = 0; i < model.descriptors.size(); ++i) {
                vector<vector<DMatch>> knnMatches;
                matcher.knnMatch(model.descriptors[i], sceneDesc, knnMatches, 2);

                vector<Point2f> objPts, scenePts;
                vector<DMatch> goodMatches;

                // Filter good matches using ratio test
                for (auto &m : knnMatches) {
                    if (m.size() == 2 && m[0].distance < MATCH_RATIO_THRESHOLD * m[1].distance) {
                        objPts.push_back(model.keypoints[i][m[0].queryIdx].pt);
                        scenePts.push_back(sceneKP[m[0].trainIdx].pt);
                        goodMatches.push_back(m[0]);
                    }
                }

                if (goodMatches.size() < MIN_INLIERS) continue;

                // Find homography between matched points
                Mat inlierMask;
                Mat H = findHomography(objPts, scenePts, RANSAC, RANSAC_THRESHOLD, inlierMask);
                if (H.empty()) continue;

                if (int inlierCount = countNonZero(inlierMask); inlierCount < MIN_INLIERS) continue;

                // Validate homography determinant
                if (double detH = fabs(determinant(H)); detH < HOMOGRAPHY_DET_THRESHOLD || detH > HOMOGRAPHY_DET_UPPER_THRESHOLD) continue;

                // Store inlier points for clustering later
                vector<Point2f> currentScenePts;
                for (size_t j = 0; j < scenePts.size(); ++j) {
                    if (inlierMask.at<uchar>(static_cast<int>(j))) {
                        currentScenePts.push_back(scenePts[j]);
                    }
                }
                scalePoints(currentScenePts);  // Adjust points for scale
                allUnfilteredScenePts.insert(allUnfilteredScenePts.end(), currentScenePts.begin(), currentScenePts.end());
            }
        };

        // Process the scene at different scales for better object detection accuracy
        vector<float> scales = {0.7f, 0.85f, 1.0f, 1.15f, 1.3f};
        for (float scale : scales) {
            Mat scaledScene;
            resize(preprocessedScene, scaledScene, Size(), scale, scale);

            vector<KeyPoint> scaledKP;
            Mat scaledDesc;
            detector->detectAndCompute(scaledScene, noArray(), scaledKP, scaledDesc);

            detectAtScale(scaledScene, scaledKP, scaledDesc, scale);
        }

        // Clustering detected points to group related matches
        if (!allUnfilteredScenePts.empty()) {
            vector<vector<Point2f>> pointClusters;
            unordered_set<size_t> unassignedPoints;
            for (size_t i = 0; i < allUnfilteredScenePts.size(); ++i) {
                unassignedPoints.insert(i);
            }

            // Perform clustering using distance threshold
            while (!unassignedPoints.empty()) {
                size_t startIdx = *unassignedPoints.begin();
                unassignedPoints.erase(startIdx);

                queue<size_t> toExplore;
                toExplore.push(startIdx);
                vector<Point2f> currentCluster;
                currentCluster.push_back(allUnfilteredScenePts[startIdx]);

                while (!toExplore.empty()) {
                    size_t currentIdx = toExplore.front();
                    toExplore.pop();

                    vector<size_t> toRemove;
                    for (size_t otherIdx : unassignedPoints) {
                        if (float dist = static_cast<float>(norm(allUnfilteredScenePts[currentIdx] - allUnfilteredScenePts[otherIdx])); dist <= CLUSTER_DISTANCE_THRESHOLD) {
                            currentCluster.push_back(allUnfilteredScenePts[otherIdx]);
                            toExplore.push(otherIdx);
                            toRemove.push_back(otherIdx);
                        }
                    }
                    for (size_t idx : toRemove) {
                        unassignedPoints.erase(idx);
                    }
                }

                if (currentCluster.size() >= MIN_POINTS_PER_CLUSTER) {
                    pointClusters.push_back(currentCluster);
                } else {
                    discardedPtsGlobal.insert(discardedPtsGlobal.end(), currentCluster.begin(), currentCluster.end());
                }
            }

            // Create bounding boxes around the detected clusters
            if (!pointClusters.empty()) {
                vector<Rect> clusterBoxes;
                for (const auto& cluster : pointClusters) {
                    Rect box = boundingRect(cluster);  // Get bounding box for each cluster

                    // Calculate mean distance and standard deviation for dynamic margin adjustment
                    float meanDist = 0.0f;
                    vector<float> distances;

                    for (size_t i = 0; i < cluster.size(); ++i) {
                        for (size_t j = i + 1; j < cluster.size(); ++j) {
                            float dist = static_cast<float>(norm(cluster[i] - cluster[j]));
                            distances.push_back(dist);
                            meanDist += dist;
                        }
                    }

                    // Calculate the standard deviation of distances for dynamic margin
                    if (!distances.empty()) {
                        meanDist /= static_cast<float>(distances.size());
                    }

                    float variance = 0.0f;
                    for (const float& dist : distances) {
                        variance += static_cast<float>(pow(dist - meanDist, 2));
                    }
                    float stdDev = sqrt(variance / static_cast<float>(distances.size()));

                    // Apply dynamic margin to the bounding box
                    float margin = stdDev * DYNAMIC_MARGIN;
                    box.x -= static_cast<int>(margin);
                    box.y -= static_cast<int>(margin);
                    box.width += static_cast<int>(2 * margin);
                    box.height += static_cast<int>(2 * margin);

                    clusterBoxes.push_back(box);
                }

                // Merge overlapping bounding boxes
                vector<Rect> mergedBoxes;
                vector<bool> processed(clusterBoxes.size(), false);

                for (size_t i = 0; i < clusterBoxes.size(); ++i) {
                    if (!processed[i]) {
                        vector<Rect> boxGroup;
                        queue<size_t> toProcess;
                        toProcess.push(i);
                        processed[i] = true;

                        while (!toProcess.empty()) {
                            size_t current = toProcess.front();
                            toProcess.pop();
                            boxGroup.push_back(clusterBoxes[current]);

                            for (size_t j = 0; j < clusterBoxes.size(); ++j) {
                                if (!processed[j]) {
                                    Point2f center1(static_cast<float>(clusterBoxes[current].x) + static_cast<float>(clusterBoxes[current].width) / 2.0f,
                                                     static_cast<float>(clusterBoxes[current].y) + static_cast<float>(clusterBoxes[current].height) / 2.0f);
                                    Point2f center2(static_cast<float>(clusterBoxes[j].x) + static_cast<float>(clusterBoxes[j].width) / 2.0f,
                                                     static_cast<float>(clusterBoxes[j].y) + static_cast<float>(clusterBoxes[j].height) / 2.0f);

                                    if (norm(center1 - center2) <= BOX_MERGE_DISTANCE) {
                                        processed[j] = true;
                                        toProcess.push(j);
                                    }
                                }
                            }
                        }

                        // Create merged bounding box for the cluster group
                        if (!boxGroup.empty()) {
                            int min_x = INT_MAX, min_y = INT_MAX;
                            int max_x = INT_MIN, max_y = INT_MIN;
                            for (const auto& box : boxGroup) {
                                min_x = min(min_x, box.x);
                                min_y = min(min_y, box.y);
                                max_x = max(max_x, box.x + box.width);
                                max_y = max(max_y, box.y + box.height);
                            }
                            mergedBoxes.emplace_back(min_x, min_y, max_x - min_x, max_y - min_y);
                        }
                    }
                }

                // Final filtering of boxes based on area
                for (const auto& box : mergedBoxes) {
                    int boxArea = box.width * box.height;
                    if (boxArea < MAX_BOX_AREA) {
                        cout << "Rejected box for " << model.name
                             << " - Area too large: " << boxArea
                             << " (max allowed: " << MAX_BOX_AREA << ")\n";
                        continue;
                    }
                    detections.emplace_back(box, model.name);
                }

                // Visualization: Draw clusters, merged boxes, and final detections
                Mat outputScene = scene.clone();
                RNG rng(12345);  // Random number generator for color selection

                // Draw points and boxes for visual feedback
                for (size_t i = 0; i < pointClusters.size(); ++i) {
                    Scalar color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
                    for (const auto& pt : pointClusters[i]) {
                        circle(outputScene, pt, 3, color, -1);
                    }
                    rectangle(outputScene, clusterBoxes[i], color, 2);
                }

                // Draw the final bounding boxes with area check
                for (const auto& box : mergedBoxes) {
                    int boxArea = box.width * box.height;
                    Scalar color = (boxArea < MAX_BOX_AREA) ? Scalar(0, 165, 255) : Scalar(0, 255, 0);
                    rectangle(outputScene, box, color, 3);
                }

                // Draw discarded points
                for (const auto& pt : discardedPtsGlobal) {
                    circle(outputScene, pt, 5, Scalar(0, 0, 255), -1);
                }
            }
        }
    }

    return detections;
}
