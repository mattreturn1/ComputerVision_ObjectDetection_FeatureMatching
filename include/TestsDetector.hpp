#ifndef TESTS_DETECTOR_HPP
#define TESTS_DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <string>

#include "objectModel.hpp"

std::vector<std::pair<cv::Rect, std::string> > detectObjects(
    const cv::Mat &scene,
    const std::string &sceneName,
    const std::vector<ObjectModel> &models,
    cv::Ptr<cv::Feature2D> &detector
);

#endif
