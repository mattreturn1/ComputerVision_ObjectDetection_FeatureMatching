//Mattia Cozza
#include "utils.hpp"
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;
// Save detections to file
void saveDetections(const string& filepath, const vector<pair<Rect, string>>& detections) {
    ofstream file(filepath);
    for (const auto& [box, name] : detections) {
        file << name << " "
             << box.x << " " << box.y << " "
             << box.x + box.width << " " << box.y + box.height
             << "\n";
    }
}

// Draw bounding boxes on image
void drawBoundingBoxes(Mat& image, const vector<pair<Rect, string>>& detections) {
    for (const auto& [box, name] : detections) {
        rectangle(image, box, Scalar(0, 255, 0), 2);
        putText(image, name, box.tl() + Point(5, 20),
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 0, 0), 2);
    }
}
