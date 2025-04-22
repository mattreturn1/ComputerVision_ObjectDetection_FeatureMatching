#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

using namespace cv;
using namespace std;

enum DetectorType { SIFT_DETECTOR, ORB_DETECTOR, FAST_BRIEF_DETECTOR };
Mat preprocessImage(const Mat& image) {
    Mat gray, stretched;

    // 1) Conversione in scala di grigi
    if (image.channels() == 3) {
        cvtColor(image, gray, COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }

    // 2) Contrast Stretching
    double minVal, maxVal;
    minMaxLoc(gray, &minVal, &maxVal);

    if (maxVal > minVal) {
        gray.convertTo(stretched, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
    } else {
        stretched = gray.clone(); // Nessuna variazione se immagine piatta
    }

    return stretched;
}



Ptr<Feature2D> createFeatureDetector(DetectorType type) {
    switch (type) {
        case SIFT_DETECTOR:
            return SIFT::create();
        case ORB_DETECTOR:
            return ORB::create();
        case FAST_BRIEF_DETECTOR:
            return ORB::create(); // ORB uses FAST + BRIEF
        default:
            return SIFT::create();
    }
}

int main() {
    // Load model, mask, and scene images
    Mat modelImg = imread("./data/035_power_drill/models/view_30_005_color.png", IMREAD_GRAYSCALE);
    Mat maskImg = imread("./data/035_power_drill/models/view_30_005_mask.png", IMREAD_GRAYSCALE);
    Mat sceneImg = imread("./data/035_power_drill/test_images/35_0038_002606-color.jpg", IMREAD_GRAYSCALE);

    if (modelImg.empty() || maskImg.empty() || sceneImg.empty()) {
        cout << "Error: One or more images not found!" << endl;
        return -1;
    }

    modelImg = preprocessImage(modelImg);
    sceneImg = preprocessImage(sceneImg);
    DetectorType detectorChoice = SIFT_DETECTOR;
    Ptr<Feature2D> detector = createFeatureDetector(detectorChoice);

    // Detect and compute features on the model with mask
    vector<KeyPoint> modelKeypoints;
    Mat modelDescriptors;
    detector->detectAndCompute(modelImg, maskImg, modelKeypoints, modelDescriptors);

    // Detect and compute features on the scene
    vector<KeyPoint> sceneKeypoints;
    Mat sceneDescriptors;
    detector->detectAndCompute(sceneImg, noArray(), sceneKeypoints, sceneDescriptors);

    cout << "Model keypoints: " << modelKeypoints.size() << endl;
    cout << "Scene keypoints: " << sceneKeypoints.size() << endl;

    // Matching
    BFMatcher matcher(detectorChoice == SIFT_DETECTOR ? NORM_L2 : NORM_HAMMING);
    vector<vector<DMatch>> knnMatches;
    matcher.knnMatch(modelDescriptors, sceneDescriptors, knnMatches, 2);

    // Apply Lowe's ratio test
    vector<DMatch> goodMatches;
    for (auto& m : knnMatches) {
        if (m.size() == 2 && m[0].distance < 0.75f * m[1].distance) {
            goodMatches.push_back(m[0]);
        }
    }

    cout << "Good matches found: " << goodMatches.size() << endl;

    // Draw Matches
    Mat matchImg;
    drawMatches(modelImg, modelKeypoints, sceneImg, sceneKeypoints, goodMatches, matchImg,
                Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    imshow("Good Matches", matchImg);
    waitKey(0);
    return 0;
}
