#include "Preprocessing.hpp"
#include "Filter.hpp"
#include <iostream>
#include <filesystem>
#include <unordered_map>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

// Funzione per caricare immagini, maschere e calcolare keypoints/descriptors
void computingModels(const string &basePath, vector<ObjectModel> &models, Ptr<Feature2D> &detector) {
    if (!fs::exists(basePath)) {
        cout << "Error: basePath does not exist: " << basePath << endl;
        return;
    }

    for (const fs::directory_entry &folder: fs::directory_iterator(basePath)) {
        if (!fs::is_directory(folder)) continue;

        ObjectModel model;
        model.name = folder.path().filename().string();
        string modelsFolder = folder.path().string() + "/models/";

        unordered_map<string, string> colorFiles;
        unordered_map<string, string> maskFiles;

        for (const fs::directory_entry &file: fs::directory_iterator(modelsFolder)) {
            string filename = file.path().filename().string();
            size_t colorPos = filename.find("_color");
            size_t maskPos = filename.find("_mask");

            if (colorPos != string::npos) {
                string baseName = filename.substr(0, colorPos);
                colorFiles[baseName] = file.path().string();
            } else if (maskPos != string::npos) {
                string baseName = filename.substr(0, maskPos);
                maskFiles[baseName] = file.path().string();
            }
        }

        for (const auto &[baseName, colorPath]: colorFiles) {
            Mat img = imread(colorPath, IMREAD_GRAYSCALE);
            if (img.empty()) {
                cout << "Error loading image: " << colorPath << endl;
                continue;
            }

            Mat mask;
            auto it = maskFiles.find(baseName);
            if (it != maskFiles.end()) {
                mask = imread(it->second, IMREAD_GRAYSCALE);
                if (mask.empty()) {
                    cout << "Error loading mask: " << it->second << endl;
                }
            } else {
                cout << "Mask not found for: " << baseName << endl;
            }

            Mat preprocessedScene = preprocessImage(img);

            vector<KeyPoint> kp;
            Mat des;
            detector->detectAndCompute(preprocessedScene, mask, kp, des);

            model.images.push_back(img);
            model.keypoints.push_back(kp);
            model.descriptors.push_back(des);
        }

        models.push_back(model);
    }
}
