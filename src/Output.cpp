//Francesco Campigotto

#include "Output.hpp"
#include "utils.hpp"
#include "TestsDetector.hpp"
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

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

            auto detections = detectObjects(scene, models, detector);

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
