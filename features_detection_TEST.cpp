#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

using namespace cv;
using namespace std;

enum DetectorType { SIFT_DETECTOR, ORB_DETECTOR, FAST_BRIEF_DETECTOR };

Ptr<Feature2D> createFeatureDetector(DetectorType type) {
    switch (type) {
        case SIFT_DETECTOR:
            return SIFT::create();
        case ORB_DETECTOR:
            return ORB::create();
        case FAST_BRIEF_DETECTOR:
            return ORB::create(); // ORB già usa FAST + BRIEF come base
        default:
            return SIFT::create();
    }
}

int main() {
    // Load image
    Mat img = imread("./data/004_sugar_box/models/view_0_001_color.png", IMREAD_GRAYSCALE);
    if (img.empty()) {
        cout << "Image not found!" << endl;
        return -1;
    }

    // Select detector type
    DetectorType detectorChoice = SIFT_DETECTOR;  // Cambia in ORB_DETECTOR o FAST_BRIEF_DETECTOR

    // Create detector
    Ptr<Feature2D> detector = createFeatureDetector(detectorChoice);

    // Detect keypoints and compute descriptors
    vector<KeyPoint> keypoints;
    Mat descriptors;
    detector->detectAndCompute(img, noArray(), keypoints, descriptors);

    // Draw keypoints
    Mat output;
    drawKeypoints(img, keypoints, output, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // Show result
    imshow("Keypoints", output);
    waitKey(0);
    return 0;
}
