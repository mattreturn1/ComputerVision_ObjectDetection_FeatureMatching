#include "ModelsDetector.hpp"
#include "preprocessing.hpp"
#include <iostream>
#include <filesystem>
#include <unordered_map>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

// Load object models from disk, apply preprocessing, and compute keypoints/descriptors
void processAllModelsImages(const string &basePath, vector<ObjectModel> &models, Ptr<Feature2D> &detector) {
    if (!fs::exists(basePath)) {
        cout << "Error: basePath does not exist: " << basePath << endl;
        return;
    }

    // Iterate through each subfolder in basePath (each represents a model)
    for (const fs::directory_entry &folder: fs::directory_iterator(basePath)) {
        if (!fs::is_directory(folder)) continue;

        ObjectModel model;
        model.name = folder.path().filename().string();
        string modelsFolder = folder.path().string() + "/models/";

        unordered_map<string, string> colorFiles;
        unordered_map<string, string> maskFiles;

        // Collect paths of color images and masks, grouped by base name
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

        // Process each color image with its corresponding mask (if available)
        for (const std::pair<const string, string> &entry: colorFiles) {
            const string &baseName = entry.first;
            const string &colorPath = entry.second;

            Mat img = imread(colorPath, IMREAD_GRAYSCALE);
            if (img.empty()) {
                cout << "Error loading image: " << colorPath << endl;
                continue;
            }

            // Try to load matching mask
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

            // Apply preprocessing before feature detection
            Mat preprocessedScene = preprocessImage(img);

            // Detect and compute keypoints and descriptors
            vector<KeyPoint> kp;
            Mat des;
            detector->detectAndCompute(preprocessedScene, mask, kp, des);

            // Store the processed data in the model
            model.images.push_back(img);
            model.keypoints.push_back(kp);
            model.descriptors.push_back(des);
        }

        // Save the fully constructed model
        models.push_back(model);
    }
}
