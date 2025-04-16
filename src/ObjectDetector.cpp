#include "ObjectDetector.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

void loadSyntheticViews(const string& basePath, vector<ObjectModel>& models, Ptr<Feature2D>& detector) {
    if (!fs::exists(basePath)) {
        cout << "Error: basePath does not exist: " << basePath << endl;
        return;
    }

    for (const auto& folder : fs::directory_iterator(basePath)) {
        if (!fs::is_directory(folder)) continue;

        ObjectModel model;
        model.name = folder.path().filename().string();
        string modelsFolder = folder.path().string() + "/models/";

        for (const auto& file : fs::directory_iterator(modelsFolder)) {
            string filename = file.path().filename().string();

            if (filename.find("_mask") != string::npos) continue;

            Mat img = imread(file.path().string(), IMREAD_GRAYSCALE);
            if (img.empty()) {
                cout << "Error loading: " << file.path() << endl;
                continue;
            }

            vector<KeyPoint> kp;
            Mat des;
            detector->detectAndCompute(img, noArray(), kp, des);

            model.images.push_back(img);
            model.keypoints.push_back(kp);
            model.descriptors.push_back(des);

            cout << "Loaded: " << model.name << " | " << file.path().filename()
                 << " | Keypoints: " << kp.size() << endl;
        }

        models.push_back(model);
    }
}

vector<pair<Rect, string>> detectObjects(const Mat& scene, const vector<ObjectModel>& models, Ptr<Feature2D>& detector, DetectorType type) {
    vector<pair<Rect, string>> detections;
    vector<KeyPoint> sceneKP;
    Mat sceneDesc;

    detector->detectAndCompute(scene, noArray(), sceneKP, sceneDesc);
    cout << "Scene keypoints: " << sceneKP.size() << endl;

    BFMatcher matcher(type == ORB_DETECTOR || type == FAST_BRIEF_DETECTOR ? NORM_HAMMING : NORM_L2);

    for (const auto& model : models) {
        for (size_t i = 0; i < model.descriptors.size(); ++i) {
            vector<vector<DMatch>> knnMatches;
            matcher.knnMatch(model.descriptors[i], sceneDesc, knnMatches, 2);

            vector<Point2f> objPoints, scenePoints;
            for (const auto& m : knnMatches) {
                if (m.size() == 2 && m[0].distance < 0.75 * m[1].distance) {
                    objPoints.push_back(model.keypoints[i][m[0].queryIdx].pt);
                    scenePoints.push_back(sceneKP[m[0].trainIdx].pt);
                }
            }

            cout << model.name << " matches: " << objPoints.size() << endl;

            if (objPoints.size() >= 4) {
                Mat H = findHomography(objPoints, scenePoints, RANSAC);
                if (!H.empty()) {
                    vector<Point2f> corners = { {0,0}, {static_cast<float>(model.images[i].cols),0},
                                                {static_cast<float>(model.images[i].cols), static_cast<float>(model.images[i].rows)},
                                                {0, static_cast<float>(model.images[i].rows)} };
                    vector<Point2f> projected;
                    perspectiveTransform(corners, projected, H);

                    float minX = scene.cols, minY = scene.rows, maxX = 0, maxY = 0;
                    for (const auto& p : projected) {
                        minX = min(minX, p.x);
                        minY = min(minY, p.y);
                        maxX = max(maxX, p.x);
                        maxY = max(maxY, p.y);
                    }

                    Rect box(Point2f(minX, minY), Point2f(maxX, maxY));
                    if (box.area() > 100) {
                        detections.emplace_back(box, model.name);
                        cout << "Detected: " << model.name << " | Box: [" << box.x << "," << box.y << "," << box.x + box.width << "," << box.y + box.height << "]" << endl;
                    }
                } else {
                    cout << "Homography failed for: " << model.name << endl;
                }
            } else {
                cout << "Not enough matches for: " << model.name << endl;
            }
        }
    }

    return detections;
}

void saveDetections(const string& filepath, const vector<pair<Rect, string>>& detections) {
    ofstream file(filepath);
    int id = 0;
    for (const auto& [box, name] : detections) {
        file << id++ << "_" << name << " " << box.x << " " << box.y << " " << box.x + box.width << " " << box.y + box.height << " 1\n";
    }
}

void drawBoundingBoxes(Mat& image, const vector<pair<Rect, string>>& detections) {
    for (const auto& [box, name] : detections) {
        rectangle(image, box, Scalar(0, 255, 0), 2);
        putText(image, name, box.tl() + Point(5, 20), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 0, 0), 2);
    }
}

