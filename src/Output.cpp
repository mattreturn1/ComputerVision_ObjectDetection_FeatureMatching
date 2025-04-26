//Francesco Campigotto

#include "Output.hpp"
#include "TestsDetector.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;
// Save detections to file
//TODO CONTROL id
void saveDetections(const string& filepath, const vector<pair<Rect, string>>& detections) {
    ofstream file(filepath);
    for (const auto& [box, name] : detections) {
        file << name << " "
             << box.x << " " << box.y << " "
             << box.x + box.width << " " << box.y + box.height
             << "\n";
    }
}

// Draw bounding boxes on image
void drawBoundingBoxes(Mat& image, const vector<pair<Rect, string>>& detections) {
    for (const auto& [box, name] : detections) {
        rectangle(image, box, Scalar(0, 255, 0), 2);
        putText(image, name, box.tl() + Point(5, 20),
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 0, 0), 2);
    }
}

// Process all test images, detect objects, save results
void processAllTestImages(
    const string& basePath,
    const vector<ObjectModel>& models,
    Ptr<Feature2D>& detector
) {
    vector<string> objectFolders = {"004_sugar_box","006_mustard_bottle","035_power_drill"};

    if (!fs::exists("./output/")) fs::create_directory("./output/");

    for (const auto& folder : objectFolders) {
        string testImagesPath = basePath + folder + "/test_images/";
        if (!fs::exists(testImagesPath)) continue;

        for (const auto& entry : fs::directory_iterator(testImagesPath)) {

            if (!fs::exists("./output/"+folder)) fs::create_directory("./output/"+folder);

            if (entry.path().extension() != ".jpg" && entry.path().extension() != ".png")
                continue;

            Mat scene = imread(entry.path().string(), IMREAD_COLOR);
            if (scene.empty()) {
                cerr << "Errore: Impossibile caricare " << entry.path().string() << endl;
                continue;
            }


            string sceneName = entry.path().stem().string();
            cout << "\nProcessing scene: " << sceneName << endl;

            auto detections = detectObjects(scene, sceneName, models, detector);

            string resultFile = "./output/" + folder + "/" + sceneName + "_results.txt";
            saveDetections(resultFile, detections);

            Mat colorScene = scene.clone();
            drawBoundingBoxes(colorScene, detections);
            string outImg = "./output/" + sceneName + "_output.png";
            imwrite(outImg, colorScene);


            cout << "Results saved for " << sceneName << endl;
        }
    }
}
