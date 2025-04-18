#include <opencv2/opencv.hpp>
#include "FeatureFactory.hpp"
#include "ObjectDetector.hpp"
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;
using namespace std;
using namespace cv;

//TODO PASS AS ARGS

int main() {
    DetectorType detectorChoice = SIFT_DETECTOR;
    Ptr<Feature2D> detector = createFeatureDetector(detectorChoice);

    vector<ObjectModel> models;
    string dataPath = "./data/";
    loadSyntheticViews(dataPath, models, detector);

    Mat scene = imread("./data/004_sugar_box/test_images/4_0001_000121-color.jpg", IMREAD_GRAYSCALE);
    if (scene.empty()) {
        cout << "Error: Could not load test image!" << endl;
        return -1;
    }

    if (!fs::exists("./output/")) fs::create_directory("./output/");

    auto detections = detectObjects(scene, models, detector, detectorChoice);
    saveDetections("./output/results.txt", detections);

    Mat colorScene;
    cvtColor(scene, colorScene, COLOR_GRAY2BGR);
    drawBoundingBoxes(colorScene, detections);
    imwrite("./output/output_image.png", colorScene);

    cout << "Detection complete. Results saved." << endl;
    return 0;
}
