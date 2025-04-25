//Mattia Cozza

#include "preprocessing.hpp"

// Preprocess image: convert to grayscale if it's a color image
Mat preprocessImage(const Mat &image) {
    Mat gray, filteredImage;

    // Convert to grayscale only if the image has 3 channels (color)
    if (image.channels() == 3) {
        cvtColor(image, gray, COLOR_BGR2GRAY);
    } else {
        gray = image.clone(); // Already grayscale
    }

    return gray;
}
