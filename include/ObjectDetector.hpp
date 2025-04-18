#ifndef OBJECT_DETECTOR_HPP
#define OBJECT_DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <string>

enum DetectorType {
    SIFT_DETECTOR,
    ORB_DETECTOR,
    FAST_BRIEF_DETECTOR,
    BRISK_DETECTOR,
    AKAZE_DETECTOR,
    KAZE_DETECTOR
};

struct ObjectModel {
    std::string name;
    std::vector<cv::Mat> images;
    std::vector<std::vector<cv::KeyPoint>> keypoints;
    std::vector<cv::Mat> descriptors;
};

std::vector<std::pair<cv::Rect, std::string>> detectObjects(const cv::Mat& scene, const std::vector<ObjectModel>& models, cv::Ptr<cv::Feature2D>& detector, DetectorType type);

void saveDetections(const std::string& filepath, const std::vector<std::pair<cv::Rect, std::string>>& detections);

void drawBoundingBoxes(cv::Mat& image, const std::vector<std::pair<cv::Rect, std::string>>& detections);

// NEW FUNCTION
void processAllTestImages(const std::string& basePath, const std::vector<ObjectModel>& models, cv::Ptr<cv::Feature2D>& detector, DetectorType type);

#endif
