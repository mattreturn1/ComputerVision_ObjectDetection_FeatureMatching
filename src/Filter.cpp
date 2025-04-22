#include "Filter.hpp"

// Funzione per applicare i filtri di pre-elaborazione con riduzione ombre
Mat preprocessImage(const Mat& image) {
    Mat gray;

    // 1) Conversione in scala di grigi
    if (image.channels() == 3) {
        cvtColor(image, gray, COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }

    return gray;
}

