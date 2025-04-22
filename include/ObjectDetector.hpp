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
    std::vector<std::vector<cv::KeyPoint> > keypoints;
    std::vector<cv::Mat> descriptors;
};

std::vector<std::pair<cv::Rect, std::string> > detectObjects(
    const cv::Mat &scene,
    const std::string &sceneName,
    const std::vector<ObjectModel> &models,
    cv::Ptr<cv::Feature2D> &detector,
    DetectorType type
);

#endif
