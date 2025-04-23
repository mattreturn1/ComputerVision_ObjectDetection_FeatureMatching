#include "Filter.hpp"
// Funzione per applicare i filtri di pre-elaborazione
Mat preprocessImage(const Mat& image) {
    Mat gray, filteredImage;

    if (image.channels() == 3) {
        cvtColor(image, gray, COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    /*
    // Opzionale: aumento contrasto locale
    equalizeHist(gray, filteredImage);
    // Opzionale: leggero sharpening
    Mat kernel = (Mat_<float>(3,3) <<
        0, -1,  0,
       -1,  5, -1,
        0, -1,  0);
    filter2D(filteredImage, filteredImage, gray.depth(), kernel);
    */
    return gray;

}



