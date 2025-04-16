#ifndef FEATURE_FACTORY_HPP
#define FEATURE_FACTORY_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "ObjectDetector.hpp"

cv::Ptr<cv::Feature2D> createFeatureDetector(DetectorType type);

#endif
