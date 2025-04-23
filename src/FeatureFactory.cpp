#include "FeatureFactory.hpp"
//TODO DELETE AND MAINTAIN ONLY SIFT
cv::Ptr<cv::Feature2D> createFeatureDetector(DetectorType type) {
    switch (type) {
        case SIFT_DETECTOR:
            //return cv::SIFT::create(0,5,0.02,15,1.6);
            return cv::SIFT::create();

        case ORB_DETECTOR:
            return cv::ORB::create();
        case BRISK_DETECTOR:
            return cv::BRISK::create();
        case AKAZE_DETECTOR:
            return cv::AKAZE::create();
        case KAZE_DETECTOR:
            return cv::KAZE::create();
        default:
            return cv::SIFT::create();
    }
}
