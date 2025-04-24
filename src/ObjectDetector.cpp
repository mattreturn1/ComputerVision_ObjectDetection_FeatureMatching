#include "ObjectDetector.hpp"
#include "Filter.hpp"
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;


/* // FUNZIONE Mattia
vector<pair<Rect, string>> detectObjects(
    const Mat &scene,
    const string &sceneName,
    const vector<ObjectModel> &models,
    Ptr<Feature2D> &detector,
    DetectorType type
) {
    const float MATCH_RATIO_THRESHOLD = 0.75f;
    const int MIN_INLIERS = 6;
    const double RANSAC_THRESHOLD = 5.0;
    const float HOMOGRAPHY_DET_THRESHOLD = 0.1;
    const float HOMOGRAPHY_DET_UPPER_THRESHOLD = 10.0;
    const float MAX_POINT_DISTANCE = 50.0f;

    Mat preprocessedScene = preprocessImage(scene);
    vector<pair<Rect, string>> detections;
    vector<KeyPoint> sceneKP;
    Mat sceneDesc;
    detector->detectAndCompute(preprocessedScene, noArray(), sceneKP, sceneDesc);

    BFMatcher matcher((type == ORB_DETECTOR || type == FAST_BRIEF_DETECTOR) ? NORM_HAMMING : NORM_L2);

    string matchDir = "./output/matches/";
    if (!fs::exists(matchDir)) fs::create_directories(matchDir);

    for (const auto &model : models) {
        vector<Point2f> allUnfilteredScenePts;
        vector<Point2f> discardedPtsGlobal;
        bool isPowerDrill = (model.name.find("power_drill") != string::npos);
        auto detectAtScale = [&](const Mat& sceneImg, const vector<KeyPoint>& kp, const Mat& desc, float scale = 1.0f) {
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

                Mat inlierMask;
                Mat H = findHomography(objPts, scenePts, RANSAC, RANSAC_THRESHOLD, inlierMask);
                if (H.empty()) continue;

                int inlierCount = countNonZero(inlierMask);
                if (inlierCount < MIN_INLIERS) continue;

                double detH = fabs(determinant(H));
                if (detH < HOMOGRAPHY_DET_THRESHOLD || detH > HOMOGRAPHY_DET_UPPER_THRESHOLD) continue;

                // Accumula gli inlier di questa view (senza filtraggio)
                for (size_t j = 0; j < scenePts.size(); ++j) {
                    if (inlierMask.at<uchar>(j)) {
                        allUnfilteredScenePts.push_back(scenePts[j]);
                    }
                }

                // Disegna match base
                Mat matchImg;
                drawMatches(
                    model.images[i], model.keypoints[i],
                    scene, sceneKP,
                    goodMatches,
                    matchImg,
                    Scalar::all(-1), Scalar::all(-1),
                    vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
                );

                string outPath = matchDir + sceneName + "_" + model.name + "_view" + to_string(i);
                if (scale != 1.0f) outPath += "_scale" + to_string(scale);
                outPath += ".png";
                imwrite(outPath, matchImg);
                cout << "Saved match image: " << outPath << endl;
            }
        };
        // Multi-scale handling for power drill
        if (isPowerDrill) {
            vector<float> scales = {0.7f, 0.85f, 1.0f, 1.15f, 1.3f};
            for (float scale : scales) {
                Mat scaledScene;
                resize(preprocessedScene, scaledScene, Size(), scale, scale);

                vector<KeyPoint> scaledKP;
                Mat scaledDesc;
                detector->detectAndCompute(scaledScene, noArray(), scaledKP, scaledDesc);

                detectAtScale(scaledScene, scaledKP, scaledDesc, scale);
            }
        } else {
            // Regular detection
            detectAtScale(preprocessedScene, sceneKP, sceneDesc);
        }
        // Calcolo centroide globale e filtraggio
        vector<Point2f> finalFilteredPts;
        if (!allUnfilteredScenePts.empty()) {
            Point2f globalMean(0, 0);
            for (const auto &pt : allUnfilteredScenePts) globalMean += pt;
            globalMean *= (1.0f / allUnfilteredScenePts.size());

            for (const auto &pt : allUnfilteredScenePts) {
                if (norm(pt - globalMean) <= MAX_POINT_DISTANCE) {
                    finalFilteredPts.push_back(pt);
                } else {
                    discardedPtsGlobal.push_back(pt);
                }
            }
        }

        // Se ci sono punti validi dopo il filtro globale, crea detection
        if (!finalFilteredPts.empty()) {
            Rect mergedBox = boundingRect(finalFilteredPts);
            detections.emplace_back(mergedBox, model.name);

            Mat outputScene = scene.clone();
            rectangle(outputScene, mergedBox, Scalar(0, 255, 0), 2);

            for (const auto &pt : finalFilteredPts)
                circle(outputScene, pt, 3, Scalar(0, 255, 0), -1);  // green = valid

            for (const auto &pt : discardedPtsGlobal)
                circle(outputScene, pt, 5, Scalar(0, 0, 255), -1);  // red = scartati

            string outPath = matchDir + sceneName + "_" + model.name + "_final_detection.png";
            imwrite(outPath, outputScene);
            cout << "Saved global detection image: " << outPath << endl;
        }
    }

    return detections;
}
*/
/*
// FUNZIONE Michele
vector<pair<Rect, string>> detectObjects(
    const Mat &scene,
    const string &sceneName,
    const vector<ObjectModel> &models,
    Ptr<Feature2D> &detector,
    DetectorType type
) {
    vector<pair<Rect, string>> detections;

    Mat preprocessedScene = preprocessImage(scene);

    // Detect keypoints and compute descriptors in the preprocessed scene
    vector<KeyPoint> sceneKP;
    Mat sceneDesc;
    detector->detectAndCompute(preprocessedScene, noArray(), sceneKP, sceneDesc);  // Use the mask here too

    BFMatcher matcher(
        (type == ORB_DETECTOR || type == FAST_BRIEF_DETECTOR) ? NORM_HAMMING : NORM_L2
    );

    string matchDir = "./output/matches/";
    if (!fs::exists(matchDir)) fs::create_directories(matchDir);

    for (const auto &model: models) {
        // Object-specific parameters - modify as needed for power drill
        float MATCH_RATIO_THRESHOLD = 0.75f;
        int MIN_INLIERS = 4;
        double RANSAC_THRESHOLD = 5.0;
        float HOMOGRAPHY_DET_THRESHOLD = 0.1;
        float HOMOGRAPHY_DET_UPPER_THRESHOLD = 10.0;
        float MAX_POINT_DISTANCE = 100.0f;

        bool isPowerDrill = (model.name.find("power_drill") != string::npos);

        vector<Rect> candidateBoxes;

        // Core detection logic
        auto detectAtScale = [&](const Mat& sceneImg, const vector<KeyPoint>& kp, const Mat& desc, float scale = 1.0f) {
            for (size_t i = 0; i < model.descriptors.size(); ++i) {
                vector<vector<DMatch>> knnMatches;
                matcher.knnMatch(model.descriptors[i], desc, knnMatches, 2);

                vector<Point2f> objPts, scenePts;
                vector<DMatch> goodMatches;
                for (auto &m: knnMatches) {
                    if (m.size() == 2 && m[0].distance < MATCH_RATIO_THRESHOLD * m[1].distance) {
                        objPts.push_back(model.keypoints[i][m[0].queryIdx].pt);
                        scenePts.push_back(kp[m[0].trainIdx].pt);
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
                    Point2f mean(0, 0);
                    for (const auto &pt: inlierScenePts) mean += pt;
                    mean *= (1.0f / inlierScenePts.size());

                    vector<Point2f> filteredPts;
                    for (const auto &pt: inlierScenePts) {
                        if (norm(pt - mean) <= MAX_POINT_DISTANCE) {
                            filteredPts.push_back(pt);
                        }
                    }

                    if (!filteredPts.empty()) {
                        Rect box = boundingRect(filteredPts);

                        if (scale != 1.0f) {
                            box.x = static_cast<int>(box.x / scale);
                            box.y = static_cast<int>(box.y / scale);
                            box.width = static_cast<int>(box.width / scale);
                            box.height = static_cast<int>(box.height / scale);
                        }

                        candidateBoxes.push_back(box);

                        Mat matchImg;
                        drawMatches(
                            model.images[i], model.keypoints[i],
                            sceneImg, kp,
                            goodMatches,
                            matchImg,
                            Scalar::all(-1), Scalar::all(-1),
                            vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
                        );
                        string outPath = matchDir + sceneName + "_" + model.name + "_view" + to_string(i);
                        if (scale != 1.0f) outPath += "_scale" + to_string(scale);
                        outPath += ".png";
                        imwrite(outPath, matchImg);
                        cout << "Saved match image: " << outPath << endl;
                    }
                }
            }
        };

        // Multi-scale handling for power drill
        if (isPowerDrill) {
            vector<float> scales = {0.7f, 0.85f, 1.0f, 1.15f, 1.3f};
            for (float scale : scales) {
                Mat scaledScene;
                resize(preprocessedScene, scaledScene, Size(), scale, scale);

                vector<KeyPoint> scaledKP;
                Mat scaledDesc;
                detector->detectAndCompute(scaledScene, noArray(), scaledKP, scaledDesc);

                detectAtScale(scaledScene, scaledKP, scaledDesc, scale);
            }
        } else {
            // Regular detection
            detectAtScale(preprocessedScene, sceneKP, sceneDesc);
        }

        // Box merging
        if (!candidateBoxes.empty()) {
            if (isPowerDrill && candidateBoxes.size() > 1) {
                Rect largestBox = candidateBoxes[0];
                for (size_t j = 1; j < candidateBoxes.size(); ++j) {
                    if (candidateBoxes[j].area() > largestBox.area()) {
                        largestBox = candidateBoxes[j];
                    }
                }
                detections.emplace_back(largestBox, model.name);
            } else {
                Rect mergedBox = candidateBoxes[0];
                for (size_t j = 1; j < candidateBoxes.size(); ++j) {
                    mergedBox |= candidateBoxes[j];
                }
                detections.emplace_back(mergedBox, model.name);
            }

            cout << "Detected " << model.name << " in " << sceneName
                 << " merged over " << candidateBoxes.size() << " hypotheses" << endl;
        }
    }

    return detections;
}
*/

/*//Merge version
vector<pair<Rect, string>> detectObjects(
    const Mat &scene,
    const string &sceneName,
    const vector<ObjectModel> &models,
    Ptr<Feature2D> &detector,
    DetectorType type
) {
    const float MATCH_RATIO_THRESHOLD = 0.75f;
    const int MIN_INLIERS = 4;
    const double RANSAC_THRESHOLD = 4.0;
    const float HOMOGRAPHY_DET_THRESHOLD = 0.1;
    const float HOMOGRAPHY_DET_UPPER_THRESHOLD = 10.0;
    const float MAX_POINT_DISTANCE = 100.0f;
    const float MIN_BBOX_AREA = 100.0f;


    Mat preprocessedScene = preprocessImage(scene);
    vector<pair<Rect, string>> detections;

    string matchDir = "./output/matches/";
    if (!fs::exists(matchDir)) fs::create_directories(matchDir);

    BFMatcher matcher((type == ORB_DETECTOR || type == FAST_BRIEF_DETECTOR) ? NORM_HAMMING : NORM_L2);

    for (const auto &model : models) {
        vector<Point2f> allUnfilteredScenePts;
        vector<Point2f> discardedPtsGlobal;
        bool isPowerDrill = (model.name.find("power_drill") != string::npos);

        // Core detection logic with multi-scale support
        auto detectAtScale = [&](const Mat& sceneImg, const vector<KeyPoint>& kp, const Mat& desc, float scale = 1.0f) {
            vector<KeyPoint> sceneKP = kp;
            Mat sceneDesc = desc;

            // Scale points back to original coordinates if needed
            auto scalePoints = [scale](vector<Point2f>& points) {
                if (scale != 1.0f) {
                    for (auto& pt : points) {
                        pt.x /= scale;
                        pt.y /= scale;
                    }
                }
            };

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

                Mat inlierMask;
                Mat H = findHomography(objPts, scenePts, RANSAC, RANSAC_THRESHOLD, inlierMask);
                if (H.empty()) continue;

                int inlierCount = countNonZero(inlierMask);
                if (inlierCount < MIN_INLIERS) continue;

                double detH = fabs(determinant(H));
                if (detH < HOMOGRAPHY_DET_THRESHOLD || detH > HOMOGRAPHY_DET_UPPER_THRESHOLD) continue;

                // Scale points back to original coordinates before accumulating
                vector<Point2f> currentScenePts;
                for (size_t j = 0; j < scenePts.size(); ++j) {
                    if (inlierMask.at<uchar>(j)) {
                        currentScenePts.push_back(scenePts[j]);
                    }
                }
                scalePoints(currentScenePts);
                allUnfilteredScenePts.insert(allUnfilteredScenePts.end(), currentScenePts.begin(), currentScenePts.end());

                // Draw matches for this view
                Mat matchImg;
                drawMatches(
                    model.images[i], model.keypoints[i],
                    sceneImg, sceneKP,
                    goodMatches,
                    matchImg,
                    Scalar::all(-1), Scalar::all(-1),
                    vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
                );

                string outPath = matchDir + sceneName + "_" + model.name + "_view" + to_string(i);
                if (scale != 1.0f) outPath += "_scale" + to_string(scale);
                outPath += ".png";
                imwrite(outPath, matchImg);
                cout << "Saved match image: " << outPath << endl;
            }
        };


        vector<float> scales = {0.7f, 0.85f, 1.0f, 1.15f, 1.3f};
        for (float scale : scales) {
            Mat scaledScene;
            resize(preprocessedScene, scaledScene, Size(), scale, scale);

            vector<KeyPoint> scaledKP;
            Mat scaledDesc;
            detector->detectAndCompute(scaledScene, noArray(), scaledKP, scaledDesc);

            detectAtScale(scaledScene, scaledKP, scaledDesc, scale);
        }

        // Global filtering and detection creation
        if (!allUnfilteredScenePts.empty()) {
            Point2f globalMean(0, 0);
            for (const auto &pt : allUnfilteredScenePts) globalMean += pt;
            globalMean *= (1.0f / allUnfilteredScenePts.size());

            vector<Point2f> finalFilteredPts;
            for (const auto &pt : allUnfilteredScenePts) {
                if (norm(pt - globalMean) <= MAX_POINT_DISTANCE) {
                    finalFilteredPts.push_back(pt);
                } else {
                    discardedPtsGlobal.push_back(pt);
                }
            }

            cout << endl << endl << finalFilteredPts.size() << endl << endl;

            if (!finalFilteredPts.empty()) {
                Rect mergedBox = boundingRect(finalFilteredPts);
                if (mergedBox.area() >= MIN_BBOX_AREA) {
                    detections.emplace_back(mergedBox, model.name);

                    Mat outputScene = scene.clone();
                    rectangle(outputScene, mergedBox, Scalar(0, 255, 0), 2);

                    for (const auto &pt : finalFilteredPts)
                        circle(outputScene, pt, 3, Scalar(0, 255, 0), -1);  // green = valid

                    for (const auto &pt : discardedPtsGlobal)
                        circle(outputScene, pt, 5, Scalar(0, 0, 255), -1);  // red = discarded

                    string outPath = matchDir + sceneName + "_" + model.name + "_final_detection.png";
                    imwrite(outPath, outputScene);
                    cout << "Saved global detection image: " << outPath << endl;
                }
            }
        }
    }

    return detections;
}*/

//CLuster version
#include "ObjectDetector.hpp"
#include "Filter.hpp"
#include <iostream>
#include <filesystem>
#include <queue>
#include <unordered_set>
#include <limits>
#include <algorithm>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

vector<pair<Rect, string>> detectObjects(
    const Mat &scene,
    const string &sceneName,
    const vector<ObjectModel> &models,
    Ptr<Feature2D> &detector,
    DetectorType type
) {
    // Configuration constants
    const float MATCH_RATIO_THRESHOLD = 0.85f;
    const int MIN_INLIERS = 4;
    const double RANSAC_THRESHOLD = 5.0;
    const float HOMOGRAPHY_DET_THRESHOLD = 0.1;
    const float HOMOGRAPHY_DET_UPPER_THRESHOLD = 10.0;
    const float CLUSTER_DISTANCE_THRESHOLD = 20.0f;
    const int MIN_POINTS_PER_CLUSTER = 15;
    const float BOX_MERGE_DISTANCE = 150.0f;
    const int MAX_BOX_AREA = 1600;

    Mat preprocessedScene = preprocessImage(scene);
    vector<pair<Rect, string>> detections;

    // Create output directory
    string matchDir = "./output/matches/";
    if (!fs::exists(matchDir)) {
        fs::create_directories(matchDir);
    }

    BFMatcher matcher((type == ORB_DETECTOR || type == FAST_BRIEF_DETECTOR) ? NORM_HAMMING : NORM_L2);

    for (const auto &model : models) {
        vector<Point2f> allUnfilteredScenePts;
        vector<Point2f> discardedPtsGlobal;

        // Multi-scale detection logic
        auto detectAtScale = [&](const Mat& sceneImg, const vector<KeyPoint>& kp, const Mat& desc, float scale = 1.0f) {
            vector<KeyPoint> sceneKP = kp;
            Mat sceneDesc = desc;

            auto scalePoints = [scale](vector<Point2f>& points) {
                if (scale != 1.0f) {
                    for (auto& pt : points) {
                        pt.x /= scale;
                        pt.y /= scale;
                    }
                }
            };

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

                Mat inlierMask;
                Mat H = findHomography(objPts, scenePts, RANSAC, RANSAC_THRESHOLD, inlierMask);
                if (H.empty()) continue;

                int inlierCount = countNonZero(inlierMask);
                if (inlierCount < MIN_INLIERS) continue;

                double detH = fabs(determinant(H));
                if (detH < HOMOGRAPHY_DET_THRESHOLD || detH > HOMOGRAPHY_DET_UPPER_THRESHOLD) continue;

                vector<Point2f> currentScenePts;
                for (size_t j = 0; j < scenePts.size(); ++j) {
                    if (inlierMask.at<uchar>(j)) {
                        currentScenePts.push_back(scenePts[j]);
                    }
                }
                scalePoints(currentScenePts);
                allUnfilteredScenePts.insert(allUnfilteredScenePts.end(), currentScenePts.begin(), currentScenePts.end());

                // Save match images
                Mat matchImg;
                drawMatches(
                    model.images[i], model.keypoints[i],
                    sceneImg, sceneKP,
                    goodMatches,
                    matchImg,
                    Scalar::all(-1), Scalar::all(-1),
                    vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
                );
                string outPath = matchDir + sceneName + "_" + model.name + "_view" + to_string(i);
                if (scale != 1.0f) outPath += "_scale" + to_string(scale);
                outPath += ".png";
                imwrite(outPath, matchImg);
            }
        };

        // Process different scales
        vector<float> scales = {0.7f, 0.85f, 1.0f, 1.15f, 1.3f};
        for (float scale : scales) {
            Mat scaledScene;
            resize(preprocessedScene, scaledScene, Size(), scale, scale);

            vector<KeyPoint> scaledKP;
            Mat scaledDesc;
            detector->detectAndCompute(scaledScene, noArray(), scaledKP, scaledDesc);

            detectAtScale(scaledScene, scaledKP, scaledDesc, scale);
        }

        // Cluster processing
        if (!allUnfilteredScenePts.empty()) {
            // Point clustering
            vector<vector<Point2f>> pointClusters;
            unordered_set<size_t> unassignedPoints;
            for (size_t i = 0; i < allUnfilteredScenePts.size(); ++i) {
                unassignedPoints.insert(i);
            }

            while (!unassignedPoints.empty()) {
                size_t startIdx = *unassignedPoints.begin();
                unassignedPoints.erase(startIdx);

                queue<size_t> toExplore;
                toExplore.push(startIdx);
                vector<Point2f> currentCluster;
                currentCluster.push_back(allUnfilteredScenePts[startIdx]);

                while (!toExplore.empty()) {
                    size_t currentIdx = toExplore.front();
                    toExplore.pop();

                    vector<size_t> toRemove;
                    for (size_t otherIdx : unassignedPoints) {
                        float dist = norm(allUnfilteredScenePts[currentIdx] - allUnfilteredScenePts[otherIdx]);
                        if (dist <= CLUSTER_DISTANCE_THRESHOLD) {
                            currentCluster.push_back(allUnfilteredScenePts[otherIdx]);
                            toExplore.push(otherIdx);
                            toRemove.push_back(otherIdx);
                        }
                    }
                    for (size_t idx : toRemove) {
                        unassignedPoints.erase(idx);
                    }
                }

                if (currentCluster.size() >= MIN_POINTS_PER_CLUSTER) {
                    pointClusters.push_back(currentCluster);
                } else {
                    discardedPtsGlobal.insert(discardedPtsGlobal.end(), currentCluster.begin(), currentCluster.end());
                }
            }

            // Create and merge bounding boxes
            if (!pointClusters.empty()) {
                vector<Rect> clusterBoxes;
                for (const auto& cluster : pointClusters) {
                    clusterBoxes.push_back(boundingRect(cluster));
                }

                // Box merging logic
                vector<Rect> mergedBoxes;
                vector<bool> processed(clusterBoxes.size(), false);

                for (size_t i = 0; i < clusterBoxes.size(); ++i) {
                    if (!processed[i]) {
                        vector<Rect> boxGroup;
                        queue<size_t> toProcess;
                        toProcess.push(i);
                        processed[i] = true;

                        while (!toProcess.empty()) {
                            size_t current = toProcess.front();
                            toProcess.pop();
                            boxGroup.push_back(clusterBoxes[current]);

                            for (size_t j = 0; j < clusterBoxes.size(); ++j) {
                                if (!processed[j]) {
                                    Point2f center1(
                                        clusterBoxes[current].x + clusterBoxes[current].width/2.0f,
                                        clusterBoxes[current].y + clusterBoxes[current].height/2.0f
                                    );
                                    Point2f center2(
                                        clusterBoxes[j].x + clusterBoxes[j].width/2.0f,
                                        clusterBoxes[j].y + clusterBoxes[j].height/2.0f
                                    );

                                    if (norm(center1 - center2) <= BOX_MERGE_DISTANCE) {
                                        processed[j] = true;
                                        toProcess.push(j);
                                    }
                                }
                            }
                        }

                        // Create merged box
                        if (!boxGroup.empty()) {
                            int min_x = INT_MAX, min_y = INT_MAX;
                            int max_x = INT_MIN, max_y = INT_MIN;
                            for (const auto& box : boxGroup) {
                                min_x = min(min_x, box.x);
                                min_y = min(min_y, box.y);
                                max_x = max(max_x, box.x + box.width);
                                max_y = max(max_y, box.y + box.height);
                            }
                            mergedBoxes.emplace_back(min_x, min_y, max_x - min_x, max_y - min_y);
                        }
                    }
                }

                // Final filtering by box area
                for (const auto& box : mergedBoxes) {
                    int boxArea = box.width * box.height;
                    if (boxArea < MAX_BOX_AREA) {
                        cout << "Rejected box for " << model.name
                             << " - Area too large: " << boxArea
                             << " (max allowed: " << MAX_BOX_AREA << ")\n";
                        continue;
                    }
                    detections.emplace_back(box, model.name);
                }

                // Visualization
                Mat outputScene = scene.clone();
                RNG rng(12345);

                // Draw clusters and boxes
                for (size_t i = 0; i < pointClusters.size(); ++i) {
                    Scalar color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
                    for (const auto& pt : pointClusters[i]) {
                        circle(outputScene, pt, 3, color, -1);
                    }
                    rectangle(outputScene, clusterBoxes[i], color, 2);
                }

                // Draw final merged boxes with area check
                for (const auto& box : mergedBoxes) {
                    int boxArea = box.width * box.height;
                    Scalar color = (boxArea < MAX_BOX_AREA) ? Scalar(0, 165, 255) : // Orange for rejected
                                                             Scalar(0, 255, 0);    // Green for valid
                    rectangle(outputScene, box, color, 3);
                }

                // Draw discarded points
                for (const auto& pt : discardedPtsGlobal) {
                    circle(outputScene, pt, 5, Scalar(0, 0, 255), -1);
                }

                string outPath = matchDir + sceneName + "_" + model.name + "_final_detection.png";
                imwrite(outPath, outputScene);
                cout << "Saved detection visualization: " << outPath << endl;
            }
        }
    }

    return detections;
}