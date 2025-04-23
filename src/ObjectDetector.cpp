#include "ObjectDetector.hpp"
#include "Filter.hpp"
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;
// Function to detect objects in a given scene image using feature matching and homography
/*
vector<pair<Rect, string> > detectObjects(
    const Mat &scene,
    const string &sceneName,
    const vector<ObjectModel> &models,
    Ptr<Feature2D> &detector,
    DetectorType type
) {
    const float MATCH_RATIO_THRESHOLD = 0.75f;
    const int MIN_INLIERS = 4;
    const double RANSAC_THRESHOLD = 5.0;
    const float HOMOGRAPHY_DET_THRESHOLD = 0.1;
    const float HOMOGRAPHY_DET_UPPER_THRESHOLD = 10.0;
    const float MAX_POINT_DISTANCE = 100.0f; // Max distance from mean for spatial filtering

    Mat preprocessedScene = preprocessImage(scene); // Grayscale conversion or preprocessing

    vector<pair<Rect, string> > detections;
    vector<KeyPoint> sceneKP;
    Mat sceneDesc;
    detector->detectAndCompute(preprocessedScene, noArray(), sceneKP, sceneDesc);

    BFMatcher matcher(
        (type == ORB_DETECTOR || type == FAST_BRIEF_DETECTOR) ? NORM_HAMMING : NORM_L2);

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
*/
/*
vector<pair<Rect, string>> detectObjects(
    const Mat &scene,
    const string &sceneName,
    const vector<ObjectModel> &models,
    Ptr<Feature2D> &detector,
    DetectorType type
) {
    const float MATCH_RATIO_THRESHOLD = 0.75f;
    const float MAX_DIST = 250.0f; // Absolute distance threshold
    const int MIN_INLIERS = 8;
    const double RANSAC_THRESHOLD = 3.0;
    const float HOMOGRAPHY_DET_THRESHOLD = 0.1;
    const float HOMOGRAPHY_DET_UPPER_THRESHOLD = 10.0;
    const float MAX_SPATIAL_DIST = 60.0f;

    Mat preprocessedScene = preprocessImage(scene);

    vector<pair<Rect, string>> detections;
    vector<KeyPoint> sceneKP;
    Mat sceneDesc;
    detector->detectAndCompute(preprocessedScene, noArray(), sceneKP, sceneDesc);

    BFMatcher matcher((type == ORB_DETECTOR || type == FAST_BRIEF_DETECTOR) ? NORM_HAMMING : NORM_L2);

    string matchDir = "./output/matches/";
    if (!fs::exists(matchDir)) fs::create_directories(matchDir);

    Mat sceneWithBoxes = scene.clone();

    for (const auto &model : models) {
        vector<Rect> candidateBoxes;

        for (size_t i = 0; i < model.descriptors.size(); ++i) {
            vector<vector<DMatch>> knnMatches;
            matcher.knnMatch(model.descriptors[i], sceneDesc, knnMatches, 2);

            vector<Point2f> objPts, scenePts;
            vector<DMatch> goodMatches;
            for (auto &m : knnMatches) {
                if (m.size() == 2 && m[0].distance < MATCH_RATIO_THRESHOLD * m[1].distance && m[0].distance < MAX_DIST) {
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

            // Spatial filtering - Only consider points that are close to each other
            Point2f center(0, 0);
            for (const auto &pt : inlierScenePts) center += pt;
            center *= (1.0f / inlierScenePts.size());

            vector<Point2f> filteredPts;
            for (const auto &pt : inlierScenePts) {
                if (norm(pt - center) <= MAX_SPATIAL_DIST) {
                    filteredPts.push_back(pt);
                }
            }

            if (filteredPts.size() < MIN_INLIERS) continue;

            // Calculate bounding box from filtered points
            vector<Point2f> modelCorners = {
                Point2f(0, 0),
                Point2f(static_cast<float>(model.images[i].cols), 0),
                Point2f(static_cast<float>(model.images[i].cols), static_cast<float>(model.images[i].rows)),
                Point2f(0, static_cast<float>(model.images[i].rows))
            };
            vector<Point2f> sceneCorners;
            perspectiveTransform(modelCorners, sceneCorners, H);

            // Only create bounding box if the points are close
            Rect candidateBox = boundingRect(filteredPts);  // Use filtered points to create bounding box
            if (candidateBox.area() > 0) {
                candidateBoxes.push_back(candidateBox);

                for (int k = 0; k < 4; ++k) {
                    line(sceneWithBoxes, sceneCorners[k], sceneCorners[(k + 1) % 4], Scalar(0, 255, 0), 2);
                }

                Mat matchImg;
                drawMatches(model.images[i], model.keypoints[i], scene, sceneKP, goodMatches,
                            matchImg, Scalar::all(-1), Scalar::all(-1), vector<char>(),
                            DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
                string matchOutPath = matchDir + sceneName + "_" + model.name + "_view" + to_string(i) + ".png";
                imwrite(matchOutPath, matchImg);
                cout << "Saved match image: " << matchOutPath << endl;
            }
        }

        if (!candidateBoxes.empty()) {
            Rect mergedBox = candidateBoxes[0];
            for (size_t j = 1; j < candidateBoxes.size(); ++j) {
                mergedBox |= candidateBoxes[j];
            }

            detections.emplace_back(mergedBox, model.name);

            rectangle(sceneWithBoxes, mergedBox, Scalar(0, 0, 255), 2);
            putText(sceneWithBoxes, model.name, mergedBox.tl() + Point(5, -10),
                    FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 2);

            cout << "Detected " << model.name << " in " << sceneName
                 << " merged over " << candidateBoxes.size() << " hypotheses" << endl;
        }
    }

    string finalScenePath = matchDir + sceneName + "_detections.png";
    imwrite(finalScenePath, sceneWithBoxes);
    cout << "Saved final scene with boxes: " << finalScenePath << endl;

    return detections;
}
*/
vector<pair<Rect, string>> detectObjects(
    const Mat &scene,
    const string &sceneName,
    const vector<ObjectModel> &models,
    Ptr<Feature2D> &detector,
    DetectorType type
) {
    const float MATCH_RATIO_THRESHOLD = 0.75f;
    const int MIN_INLIERS = 4;
    const double RANSAC_THRESHOLD = 5.0;
    const float HOMOGRAPHY_DET_THRESHOLD = 0.1;
    const float HOMOGRAPHY_DET_UPPER_THRESHOLD = 10.0;
    const float MAX_POINT_DISTANCE = 30.0f; // Max distance from mean for spatial filtering

    Mat preprocessedScene = preprocessImage(scene); // Grayscale conversion or preprocessing

    vector<pair<Rect, string>> detections;
    vector<KeyPoint> sceneKP;
    Mat sceneDesc;
    detector->detectAndCompute(preprocessedScene, noArray(), sceneKP, sceneDesc);

    BFMatcher matcher(
        (type == ORB_DETECTOR || type == FAST_BRIEF_DETECTOR) ? NORM_HAMMING : NORM_L2);

    string matchDir = "./output/matches/";
    if (!fs::exists(matchDir)) fs::create_directories(matchDir);

    for (const auto &model : models) {
        vector<Rect> candidateBoxes;
        vector<vector<DMatch>> allGoodMatches;  // To store good matches across all views of the model
        vector<Point2f> allObjPts;  // To store object points from all views
        vector<Point2f> allScenePts;  // To store scene points from all views

        for (size_t i = 0; i < model.descriptors.size(); ++i) {
            vector<vector<DMatch>> knnMatches;
            matcher.knnMatch(model.descriptors[i], sceneDesc, knnMatches, 2);

            vector<Point2f> objPts, scenePts;
            vector<DMatch> goodMatches;
            for (auto &m : knnMatches) {
                if (m.size() == 2 && m[0].distance < MATCH_RATIO_THRESHOLD * m[1].distance) {
                    objPts.push_back(model.keypoints[i][m[0].queryIdx].pt);
                    scenePts.push_back(sceneKP[m[0].trainIdx].pt);
                    goodMatches.push_back(m[0]);
                }
            }

            if (goodMatches.size() < MIN_INLIERS) continue;

            // Add the good matches of the current view to the list of all good matches
            allGoodMatches.push_back(goodMatches);
            allObjPts.insert(allObjPts.end(), objPts.begin(), objPts.end());
            allScenePts.insert(allScenePts.end(), scenePts.begin(), scenePts.end());

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
                for (const auto &pt : inlierScenePts) mean += pt;
                mean *= (1.0f / inlierScenePts.size());

                // Filter out isolated points too far from the center
                vector<Point2f> filteredPts;
                for (const auto &pt : inlierScenePts) {
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

        if (!allGoodMatches.empty()) {
            // Combine all good matches into one list
            vector<Point2f> allFilteredPts;
            for (size_t i = 0; i < allScenePts.size(); ++i) {
                allFilteredPts.push_back(allScenePts[i]);
            }

            // Draw bounding box around all matched points
            if (!allFilteredPts.empty()) {
                Rect mergedBox = boundingRect(allFilteredPts);
                detections.emplace_back(mergedBox, model.name);

                cout << "Detected " << model.name << " in " << sceneName
                     << " across " << allGoodMatches.size() << " views" << endl;

                // Optionally, draw the bounding box on the scene image
                Mat outputScene = scene.clone();
                rectangle(outputScene, mergedBox, Scalar(0, 255, 0), 2);
                string outPath = matchDir + sceneName + "_" + model.name + "_final_detection.png";
                imwrite(outPath, outputScene);
                cout << "Saved final detection image: " << outPath << endl;
            }
        }
    }

    return detections;
}
