Preprocessing:
        - For each object read all views.
        - Extract keypoints and descriptors from each view using the preferred algorithm.
        - Store the keypoints and descriptors in a std::vector or std::map for quick access.

Detection for Each Test Image:
        - Convert the input test image to grayscale.
        - Detect keypoints and compute descriptors using the preferred algorithm.
        - Loop through all views for all objects:
            - Match descriptors using BFMatcher (Brute-Force) with Euclidean distance (NORM_L2)
                - Filter matches, if enough good matches are found:
                    - Compute the Homography matrix with findHomography() using RANSAC
                    - Use perspectiveTransform() to calculate the projected bounding box coordinates in the scene
                    - Save the bounding box and draw it on the image.

Saving:
        - Write bounding box info to a .txt file using the format:
        <object_id>_<object_name> <xmin> <ymin> <xmax> <ymax> <is_present>

Output:
        - Draw the bounding boxes on the image using rectangle() and label them with putText().
