#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp> // Aggiunto per l'uso di imshow
#include <iostream>

using namespace cv;
using namespace std;

enum DetectorType { SIFT_DETECTOR, ORB_DETECTOR, FAST_BRIEF_DETECTOR };

// Funzione di preprocessing immagine
Mat preprocessImage(const Mat& image) {
    Mat gray, filteredImage;

    if (image.channels() == 3) {
        cvtColor(image, gray, COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    // Opzionale: aumento contrasto locale
    equalizeHist(gray, filteredImage);
    // Opzionale: leggero sharpening
    Mat kernel = (Mat_<float>(3,3) <<
        0, -1,  0,
       -1,  5, -1,
        0, -1,  0);
    filter2D(filteredImage, filteredImage, gray.depth(), kernel);

    return filteredImage;
}

// Funzione di preprocessing per ottenere immagine in scala di grigi
Mat preprocessImage1(const Mat& image) {
    Mat gray, filteredImage;

    if (image.channels() == 3) {
        cvtColor(image, gray, COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }

    return gray;
}

// Funzione per creare un detector
Ptr<Feature2D> createFeatureDetector(DetectorType type) {
    switch (type) {
        case SIFT_DETECTOR:
            return SIFT::create(0,5,0.01,20,1.6);
        case ORB_DETECTOR:
            return ORB::create();
        case FAST_BRIEF_DETECTOR:
            return ORB::create(); // ORB usa FAST + BRIEF
        default:
            return SIFT::create();
    }
}

// Funzione per calcolare la distanza euclidea tra due punti
float calcEuclideanDistance(const Point2f& p1, const Point2f& p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

int main() {
    // Carica immagini
    Mat modelImg = imread("./data/004_sugar_box/models/view_0_003_color.png", IMREAD_GRAYSCALE);
    Mat maskImg = imread("./data/004_sugar_box/models/view_0_003_mask.png", IMREAD_GRAYSCALE);
    Mat sceneImg = imread("./data/004_sugar_box/test_images/4_0001_000956-color.jpg", IMREAD_GRAYSCALE);

    if (modelImg.empty() || maskImg.empty() || sceneImg.empty()) {
        cout << "Error: One or more images not found!" << endl;
        return -1;
    }

    // Pre-processamento delle immagini
    modelImg = preprocessImage1(modelImg);
    sceneImg = preprocessImage(sceneImg);

    // Creazione del detector (SIFT in questo caso)
    DetectorType detectorChoice = SIFT_DETECTOR;
    Ptr<Feature2D> detector = createFeatureDetector(detectorChoice);

    // Rilevamento e calcolo delle caratteristiche nel modello con maschera
    vector<KeyPoint> modelKeypoints;
    Mat modelDescriptors;
    detector->detectAndCompute(modelImg, maskImg, modelKeypoints, modelDescriptors);

    // Rilevamento e calcolo delle caratteristiche nella scena
    vector<KeyPoint> sceneKeypoints;
    Mat sceneDescriptors;
    detector->detectAndCompute(sceneImg, noArray(), sceneKeypoints, sceneDescriptors);

    cout << "Model keypoints: " << modelKeypoints.size() << endl;
    cout << "Scene keypoints: " << sceneKeypoints.size() << endl;

    // Matching
    BFMatcher matcher(detectorChoice == SIFT_DETECTOR ? NORM_L2 : NORM_HAMMING);
    vector<vector<DMatch>> knnMatches;
    matcher.knnMatch(modelDescriptors, sceneDescriptors, knnMatches, 2);

    // Applica il test di Lowe's ratio
    vector<DMatch> goodMatches;
    for (auto& m : knnMatches) {
        if (m.size() == 2 && m[0].distance < 0.75f * m[1].distance) {
            goodMatches.push_back(m[0]);
        }
    }

    cout << "Good matches found: " << goodMatches.size() << endl;

    // Rilevamento dei contorni nell'immagine di scena
    Mat edges;
    Canny(sceneImg, edges, 50, 150); // Rilevamento dei bordi con Canny
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(edges, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    cout << "Contours found: " << contours.size() << endl;

    // Contatore per trovare il contorno con più punti di matching
    vector<int> contourMatchCount(contours.size(), 0);

    // Distanza minima tra i punti di matching da considerare "vicini"
    float minDistanceThreshold = 30.0f;  // Cambia questo valore se necessario

    // Per ogni punto di matching, verifica in quale contorno si trova e se la distanza tra i punti è abbastanza piccola
    for (auto& match : goodMatches) {
        Point scenePoint = sceneKeypoints[match.trainIdx].pt;
        Point modelPoint = modelKeypoints[match.queryIdx].pt;

        // Calcola la distanza tra i punti di matching
        float distance = calcEuclideanDistance(scenePoint, modelPoint);

        // Se la distanza è inferiore alla soglia, conta la corrispondenza
        if (distance < minDistanceThreshold) {
            // Verifica in quale contorno si trova il punto
            for (size_t i = 0; i < contours.size(); ++i) {
                if (pointPolygonTest(contours[i], scenePoint, false) >= 0) {
                    contourMatchCount[i]++;
                    break;
                }
            }
        }
    }

    // Trova l'indice del contorno con il maggior numero di match
    int maxMatchesIndex = max_element(contourMatchCount.begin(), contourMatchCount.end()) - contourMatchCount.begin();

    // Se ci sono contorni rilevati, disegna solo il contorno con il maggior numero di match in verde
    if (contours.size() > 0 && contourMatchCount[maxMatchesIndex] > 0) {
        Mat contourImg = sceneImg.clone();

        // Disegna il contorno con il colore verde
        drawContours(contourImg, contours, maxMatchesIndex, Scalar(0, 255, 0), 2); // Disegna contorno verde

        // Disegnare anche le corrispondenze
        Mat matchImg;
        drawMatches(modelImg, modelKeypoints, contourImg, sceneKeypoints, goodMatches, matchImg,
                    Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        // Mostra l'immagine con il contorno verde e le corrispondenze
        imshow("Good Matches and Contours", matchImg);
        waitKey(0); // Aggiungi questa linea per fermare il programma fino a quando non si preme un tasto
    } else {
        cout << "No matching contours found!" << endl;
    }

    return 0;
}
