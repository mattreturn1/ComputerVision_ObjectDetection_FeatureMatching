#include "ObjectDetector.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

// Function to load synthetic object views from folders and extract keypoints & descriptors
void loadSyntheticViews(const string& basePath, vector<ObjectModel>& models, Ptr<Feature2D>& detector) {
    if (!fs::exists(basePath)) {  // Check if the provided base path exists
        cout << "Error: basePath does not exist: " << basePath << endl;
        return;
    }

    // Iterate through each subdirectory (representing a model)
    for (const auto& folder : fs::directory_iterator(basePath)) {
        if (!fs::is_directory(folder)) continue;

        ObjectModel model;
        model.name = folder.path().filename().string();  // Extract model name from folder name
        string modelsFolder = folder.path().string() + "/models/";

        // Iterate through all image files in the model folder
        for (const auto& file : fs::directory_iterator(modelsFolder)) {
            string filename = file.path().filename().string();

            if (filename.find("_mask") != string::npos) continue;  // Skip mask files

            Mat img = imread(file.path().string(), IMREAD_GRAYSCALE);  // Load image in grayscale
            if (img.empty()) {  // Check for loading errors
                cout << "Error loading: " << file.path() << endl;
                continue;
            }

            vector<KeyPoint> kp;
            Mat des;
            detector->detectAndCompute(img, noArray(), kp, des);  // Detect keypoints and compute descriptors

            model.images.push_back(img);         // Store image
            model.keypoints.push_back(kp);       // Store keypoints
            model.descriptors.push_back(des);    // Store descriptors

            cout << "Loaded: " << model.name << " | " << file.path().filename()
                 << " | Keypoints: " << kp.size() << endl;
        }

        models.push_back(model);  // Add the fully-loaded model to the list
    }
}

// Function to detect objects in a given scene image using feature matching and homography
vector<pair<Rect, string>> detectObjects(const Mat& scene, const vector<ObjectModel>& models, Ptr<Feature2D>& detector, DetectorType type) {
    vector<pair<Rect, string>> detections;
    vector<KeyPoint> sceneKP;
    Mat sceneDesc;

    detector->detectAndCompute(scene, noArray(), sceneKP, sceneDesc);  // Extract keypoints and descriptors from the scene
    cout << "Scene keypoints: " << sceneKP.size() << endl;

    // Choose the appropriate distance metric based on the detector type
    BFMatcher matcher(type == ORB_DETECTOR || type == FAST_BRIEF_DETECTOR ? NORM_HAMMING : NORM_L2);

    // Loop through each model and its views
    for (const auto& model : models) {
        for (size_t i = 0; i < model.descriptors.size(); ++i) {
            vector<vector<DMatch>> knnMatches;
            matcher.knnMatch(model.descriptors[i], sceneDesc, knnMatches, 2);  // Find the 2 nearest matches for each descriptor

            vector<Point2f> objPoints, scenePoints;
            // Apply Lowe's ratio test to filter matches
            for (const auto& m : knnMatches) {
                if (m.size() == 2 && m[0].distance < 0.65 * m[1].distance) {
                    objPoints.push_back(model.keypoints[i][m[0].queryIdx].pt);
                    scenePoints.push_back(sceneKP[m[0].trainIdx].pt);
                }
            }

            cout << model.name << " matches after filtering: " << objPoints.size() << endl;

            if (objPoints.size() >= 4) {  // At least 4 matches needed to compute a homography
                Mat H = findHomography(objPoints, scenePoints, RANSAC);  // Estimate the transformation
                if (!H.empty()) {
                    double det = fabs(determinant(H));
                    // Reject extreme distortions based on determinant
                    if (det < 0.1 || det > 10) {
                        cout << "Rejected detection (invalid homography) for: " << model.name << endl;
                        continue;
                    }

                    // Define object corners in the model image
                    vector<Point2f> corners = { {0,0}, {static_cast<float>(model.images[i].cols),0},
                                                {static_cast<float>(model.images[i].cols), static_cast<float>(model.images[i].rows)},
                                                {0, static_cast<float>(model.images[i].rows)} };
                    vector<Point2f> projected;
                    perspectiveTransform(corners, projected, H);  // Map corners to the scene

                    // Compute bounding box around the projected corners
                    float minX = scene.cols, minY = scene.rows, maxX = 0, maxY = 0;
                    for (const auto& p : projected) {
                        minX = min(minX, p.x);
                        minY = min(minY, p.y);
                        maxX = max(maxX, p.x);
                        maxY = max(maxY, p.y);
                    }

                    Rect box(Point2f(minX, minY), Point2f(maxX, maxY));
                    if (box.area() > 100) {  // Ignore small detections
                        detections.emplace_back(box, model.name);
                        cout << "Detected: " << model.name << " | Box: ["
                             << box.x << "," << box.y << "," << box.x + box.width << "," << box.y + box.height << "]" << endl;
                    }
                } else {
                    cout << "Homography failed for: " << model.name << endl;
                }
            } else {
                cout << "Not enough matches for: " << model.name << endl;
            }
        }
    }

    // Remove overlapping detections (keep only the most confident one)
    vector<pair<Rect, string>> filtered;
    for (const auto& det : detections) {
        bool overlap = false;
        for (const auto& f : filtered) {
            float intersectionArea = (det.first & f.first).area();
            float minArea = min((float)det.first.area(), (float)f.first.area());
            if (intersectionArea > 0.5f * minArea) {  // Consider it overlapping if more than 50% overlap
                overlap = true;
                break;
            }
        }
        if (!overlap) filtered.push_back(det);
    }

    return filtered;  // Return final list of detections
}

// Function to save detection results to a text file
void saveDetections(const string& filepath, const vector<pair<Rect, string>>& detections) {
    ofstream file(filepath);
    int id = 0;
    for (const auto& [box, name] : detections) {
        file << id++ << "_" << name << " " << box.x << " " << box.y << " " << box.x + box.width << " " << box.y + box.height << " 1\n";
    }
}

// Function to draw bounding boxes and labels on an image
void drawBoundingBoxes(Mat& image, const vector<pair<Rect, string>>& detections) {
    for (const auto& [box, name] : detections) {
        rectangle(image, box, Scalar(0, 255, 0), 2);  // Draw green rectangle
        putText(image, name, box.tl() + Point(5, 20), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 0, 0), 2);  // Draw label
    }
}
