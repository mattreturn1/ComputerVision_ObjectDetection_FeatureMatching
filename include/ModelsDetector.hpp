//Mattia Cozza

#ifndef MODELS_DETECTOR_HPP
#define MODELS_DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <string>

#include "objectModel.hpp"

void processAllModelsImages(const std::string &basePath, std::vector<ObjectModel> &models,
                            cv::Ptr<cv::Feature2D> &detector);

#endif
