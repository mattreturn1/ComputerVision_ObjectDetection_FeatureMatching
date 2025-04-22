#include <opencv2/opencv.hpp>
#include "FeatureFactory.hpp"
#include "Preprocessing.hpp"
#include "Output.hpp"
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
    computingModels(dataPath, models, detector);

    processAllTestImages(dataPath, models, detector, detectorChoice);

    cout << "Detection complete. Results saved in ./output/ directory." << endl;
    return 0;
}
