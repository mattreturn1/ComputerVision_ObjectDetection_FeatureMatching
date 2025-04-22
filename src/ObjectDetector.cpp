#include "ObjectDetector.hpp"
#include "Filter.hpp"
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

// Function to detect objects in a given scene image using feature matching and homography
// Updated detectObjects: best match per object per scene, keep all objects
vector<pair<Rect, string> > detectObjects(
    const Mat &scene,
    const string &sceneName,
    const vector<ObjectModel> &models,
    Ptr<Feature2D> &detector,
    DetectorType type
) {
    const float MATCH_RATIO_THRESHOLD = 0.75f;
    const int MIN_INLIERS = 6;
    const double RANSAC_THRESHOLD = 3.0;
    const float HOMOGRAPHY_DET_THRESHOLD = 0.1;
    const float HOMOGRAPHY_DET_UPPER_THRESHOLD = 10.0;
    const float MAX_POINT_DISTANCE = 70.0f; // Max distance from mean for spatial filtering

    Mat preprocessedScene = preprocessImage(scene); // Grayscale conversion or preprocessing

    vector<pair<Rect, string> > detections;
    vector<KeyPoint> sceneKP;
    Mat sceneDesc;
    detector->detectAndCompute(preprocessedScene, noArray(), sceneKP, sceneDesc);

    BFMatcher matcher(
        (type == ORB_DETECTOR || type == FAST_BRIEF_DETECTOR) ? NORM_HAMMING : NORM_L2
    );

    string matchDir = "./output/matches/";
    if (!fs::exists(matchDir)) fs::create_directories(matchDir);

    for (const auto &model: models) {
        vector<Rect> candidateBoxes;

        for (size_t i = 0; i < model.descriptors.size(); ++i) {
            vector<vector<DMatch> > knnMatches;
            matcher.knnMatch(model.descriptors[i], sceneDesc, knnMatches, 2);

            vector<Point2f> objPts, scenePts;
            vector<DMatch> goodMatches;
            for (auto &m: knnMatches) {
                if (m.size() == 2 && m[0].distance < MATCH_RATIO_THRESHOLD * m[1].distance) {
                    objPts.push_back(model.keypoints[i][m[0].queryIdx].pt);
                    scenePts.push_back(sceneKP[m[0].trainIdx].pt);
                    goodMatches.push_back(m[0]);
                }
            }

            if (goodMatches.size() < MIN_INLIERS) continue;

            Mat inlierMask;
            Mat H = findHomography(objPts, scenePts, RANSAC, RANSAC_THRESHOLD, inlierMask);
            if (H.empty()) continue;

            int inlierCount = countNonZero(inlierMask);
            if (inlierCount < MIN_INLIERS) continue;

            double detH = fabs(determinant(H));
            if (detH < HOMOGRAPHY_DET_THRESHOLD || detH > HOMOGRAPHY_DET_UPPER_THRESHOLD) continue;

            vector<Point2f> inlierScenePts;
            for (size_t j = 0; j < scenePts.size(); ++j) {
                if (inlierMask.at<uchar>(j)) {
                    inlierScenePts.push_back(scenePts[j]);
                }
            }

            if (!inlierScenePts.empty()) {
                // Calculate the center of the inlier cluster
                Point2f mean(0, 0);
                for (const auto &pt: inlierScenePts) mean += pt;
                mean *= (1.0f / inlierScenePts.size());

                // Filter out isolated points too far from the center
                vector<Point2f> filteredPts;
                for (const auto &pt: inlierScenePts) {
                    if (norm(pt - mean) <= MAX_POINT_DISTANCE) {
                        filteredPts.push_back(pt);
                    }
                }

                if (!filteredPts.empty()) {
                    Rect candidateBox = boundingRect(filteredPts);
                    candidateBoxes.push_back(candidateBox);

                    Mat matchImg;
                    drawMatches(
                        model.images[i], model.keypoints[i],
                        scene, sceneKP,
                        goodMatches,
                        matchImg,
                        Scalar::all(-1), Scalar::all(-1),
                        vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
                    );
                    string outPath = matchDir + sceneName + "_" + model.name + "_view" + to_string(i) + ".png";
                    imwrite(outPath, matchImg);
                    cout << "Saved match image: " << outPath << endl;
                }
            }
        }

        if (!candidateBoxes.empty()) {
            Rect mergedBox = candidateBoxes[0];
            for (size_t j = 1; j < candidateBoxes.size(); ++j) {
                mergedBox |= candidateBoxes[j]; // Merge overlapping boxes
            }
            detections.emplace_back(mergedBox, model.name);
            cout << "Detected " << model.name << " in " << sceneName
                    << " merged over " << candidateBoxes.size() << " hypotheses" << endl;
        }
    }

    return detections;
}
