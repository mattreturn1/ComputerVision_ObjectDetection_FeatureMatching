//Mattia Cozza

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>

#include "ModelsDetector.hpp"
#include "Output.hpp"
#include "metrics.h"

namespace fs = std::filesystem;
using namespace std;
using namespace cv;

//TODO pass data_path as args

int main() {
    Ptr<Feature2D> detector = SIFT::create();

    vector<ObjectModel> models;
    const string dataPath = "./data/";
    const string outputPath = "./output/";
    processAllModelsImages(dataPath, models, detector);

    processAllTestImages(dataPath, models, detector);

    cout << "Detection complete. Results saved in ./output/ directory." << endl;

    const float meanIoU = compute_mean_intersection_over_union(dataPath, outputPath);

    const float accuracy = compute_detection_accuracy(dataPath, outputPath);

    cout << "Mean IoU: " << meanIoU << endl;
    cout << "Detection Accuracy: " << accuracy << endl;

    return 0;
}
