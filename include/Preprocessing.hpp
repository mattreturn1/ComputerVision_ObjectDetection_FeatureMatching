#ifndef PREPROCESSING_HPP
#define PREPROCESSING_HPP
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <string>
#include "ObjectDetector.hpp"

//CHECK OBJECTMODEL IF CHANGED

void computingModels(const std::string& basePath, std::vector<ObjectModel>& models, cv::Ptr<cv::Feature2D>& detector);

#endif //PREPROCESSING_HPP
