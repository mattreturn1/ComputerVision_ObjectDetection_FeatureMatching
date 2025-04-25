#ifndef OBJECTMODEL_HPP
#define OBJECTMODEL_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// Structure to store model data: name, images, keypoints, and descriptors
struct ObjectModel {
    std::string name; // Model identifier
    std::vector<cv::Mat> images; // Grayscale input images
    std::vector<std::vector<cv::KeyPoint> > keypoints; // Keypoints for each image
    std::vector<cv::Mat> descriptors; // Descriptors for each image
};

#endif
