# Object Detection Pipeline Overview

## MODULE 1 — Preprocessing (MATTIA)

**Include Parser for Train/Mask Images**

Steps:
- For each object, read all available images.
- Extract keypoints and descriptors from each view using the preferred feature extraction algorithm.
- Store the keypoints and descriptors in a `std::vector` or `std::map` for quick access.
- *(Optional)*: If masks are used for models, integrate mask-based filtering.
- *(Optional)*: Save the extracted keypoints and descriptors into `.txt` files as a map for future reuse.

---

## MODULE 2 — Detection (FRANCESCO)

**Include Parser for Test Images**

Steps for Each Test Image:
- Convert the input image to grayscale.
- Detect keypoints and compute descriptors using the preferred algorithm.
- Loop through all object views:
    - Match descriptors using `BFMatcher` (Brute-Force) with **Euclidean distance (NORM_L2)**.  
      *(Other matching strategies can be considered.)*
    - Filter matches. If enough good matches are found:
        - Compute the **Homography matrix** using `findHomography()` with **RANSAC**.
        - Use `perspectiveTransform()` to calculate the projected bounding box coordinates in the scene.

---
## (MICHELE)
## MODULE 3 — Output

**Result Export**

- Write bounding box information to a `.txt` file using the following format:
```
<object_id>_<object_name> <xmin> <ymin> <xmax> <ymax> <is_present>
```

---

## MODULE 4 — Drawing

**Visualization**

- Draw the predicted bounding boxes on the image using `rectangle()`.
- Label each bounding box using `putText()` with the object name or ID.

---

## MODULE 5 — Metrics

**Evaluation**

- **Mean Intersection over Union (mIoU)**:  
  Calculate the average IoU across all object categories.
- **Detection Accuracy**:  
  Count the number of object instances correctly recognized per category.  
  An object is considered **correctly detected (true positive)** if the predicted and ground truth bounding boxes have:
```
IoU > 0.5
```

---
### Notes

- Retrieve all **good matches** for each model of the same object and place them in **our view**.
- A test contains **all features** for **all views** for **all objects**.
- Set a **low max distance** so that the most recurring points are detected as valid boxes, while distant/noisy points are filtered out.
- **Unexpected boxes** may occur due to merges between nearby boxes of different objects (controlled by a **box border distance parameter**).
- Collect all the boxes and **remove those that are too small** (controlled by a **box area parameter**).
