SUBSTITUTE auto

MODULO 1 (MATTIA)

INCLUDE PARSER FOR TRAIN/MASK IMAGES
Preprocessing:
        - For each object read all images.
        - Extract keypoints and descriptors from each view using the preferred algorithm.
        - Store the keypoints and descriptors in a std::vector or std::map for quick access.
        - (USING MASKS FOR MODELS?)
        - (OPTIONAL)(SAVE THE KEYPOINTS AND DESCRIPTORS ON .TXT AS MAP)

MODULO 2 (FRANCESCO)

INCLUDE PARSER FOR TEST IMAGES
Detection for Each Test Image:
        - Convert the input test image to grayscale.
        - Detect keypoints and compute descriptors using the preferred algorithm.
        - Loop through all views for all objects:
            - Match descriptors using BFMatcher (Brute-Force) with Euclidean distance (NORM_L2), (AND OTHERS?)
                - Filter matches, if enough good matches are found:
                    - Compute the Homography matrix with findHomography() using RANSAC
                    - Use perspectiveTransform() to calculate the projected bounding box coordinates in the scene

(MICHELE)

MODULO 3 

Output:
        - Write bounding box info to a .txt file using the format:
        <object_id>_<object_name> <xmin> <ymin> <xmax> <ymax> <is_present>

MODULO 4

Drawing:
        - Draw the bounding boxes on the image using rectangle() and label them with putText().

MODULO 5

Metrics:
        - The mean Intersection over Union (mIoU) is the average of the IoU computed for each object category;
        - For detection accuracy, the number of object instances correctly recognized for each object category, 
        considering a true positive (object is correctly detected) if the predicted and ground truth bounding boxes have IoU>0.5).



