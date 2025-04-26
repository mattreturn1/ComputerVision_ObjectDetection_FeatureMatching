//Mattia Cozza

#ifndef UTILS_HPP
#define UTILS_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

void saveDetections(const std::string &filepath, const std::vector<std::pair<cv::Rect, std::string> > &detections);

void drawBoundingBoxes(cv::Mat &image, const std::vector<std::pair<cv::Rect, std::string> > &detections);

#endif
