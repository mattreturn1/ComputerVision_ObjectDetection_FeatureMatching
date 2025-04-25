#ifndef OUTPUT_HPP
#define OUTPUT_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <string>
#include "objectModel.hpp"

void saveDetections(const std::string &filepath, const std::vector<std::pair<cv::Rect, std::string> > &detections);

void drawBoundingBoxes(cv::Mat &image, const std::vector<std::pair<cv::Rect, std::string> > &detections);

void processAllTestImages(const std::string &basePath, const std::vector<ObjectModel> &models,
                          cv::Ptr<cv::Feature2D> &detector);

#endif
