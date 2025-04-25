//Mattia Cozza

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>

#include "ModelsDetector.hpp"
#include "Output.hpp"

namespace fs = std::filesystem;
using namespace std;
using namespace cv;

//TODO pass data_path as args

int main() {
    Ptr<Feature2D> detector = SIFT::create();

    vector<ObjectModel> models;
    const string dataPath = "./data/";
    processAllModelsImages(dataPath, models, detector);

    processAllTestImages(dataPath, models, detector);

    cout << "Detection complete. Results saved in ./output/ directory." << endl;
    return 0;
}
