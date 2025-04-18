#include "Preprocessing.hpp"
#include <iostream>
#include <filesystem>
#include <unordered_map>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

// Function to load objects images from folders and extract keypoints & descriptors
void computingModels(const string& basePath, vector<ObjectModel>& models, Ptr<Feature2D>& detector) {
    if (!fs::exists(basePath)) {
        cout << "Error: basePath does not exist: " << basePath << endl;
        return;
    }

    for (const fs::directory_entry& folder : fs::directory_iterator(basePath)) {
        if (!fs::is_directory(folder)) continue;

        ObjectModel model;
        model.name = folder.path().filename().string();
        string modelsFolder = folder.path().string() + "/models/";

        unordered_map<string, string> colorFiles;
        unordered_map<string, string> maskFiles;

        // First scan: map base names to file paths
        for (const fs::directory_entry& file : fs::directory_iterator(modelsFolder)) {
            string filename = file.path().filename().string();

            size_t colorPos = filename.find("_color");
            size_t maskPos = filename.find("_mask");

            if (colorPos != string::npos) {
                string baseName = filename.substr(0, colorPos);
                colorFiles[baseName] = file.path().string();
            }
            else if (maskPos != string::npos) {
                string baseName = filename.substr(0, maskPos);
                maskFiles[baseName] = file.path().string();
            }
        }

        // Second scan: process color images, match mask if available
        for (const auto& [baseName, colorPath] : colorFiles) {
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

            vector<KeyPoint> kp;
            Mat des;
            detector->detectAndCompute(img, mask, kp, des);

            model.images.push_back(img);
            model.keypoints.push_back(kp);
            model.descriptors.push_back(des);

            cout << "Loaded: " << model.name
                 << " | " << fs::path(colorPath).filename()
                 << " | Keypoints: " << kp.size() << endl;
        }

        models.push_back(model);
    }
}
