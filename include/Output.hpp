//Francesco Campigotto

#ifndef OUTPUT_HPP
#define OUTPUT_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <string>
#include "objectModel.hpp"

void processAllTestImages(const std::string &basePath, const std::vector<ObjectModel> &models,
                          cv::Ptr<cv::Feature2D> &detector);

#endif
