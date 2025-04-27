//Michele Brigandì

#include "metrics.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>

namespace fs = std::filesystem;

// Computes the mean IoU across all object classes in the dataset
float compute_mean_intersection_over_union(const std::string& dataset_path, const std::string& output_path,
                                           const std::string& ground_truths_path) {
    std::vector<float> object_classes_iou;

    for (const fs::directory_entry& object_class : fs::directory_iterator(dataset_path)) {
        if (object_class.is_directory()) {
            fs::path ground_truth_path = object_class.path() / ground_truths_path;
            fs::path prediction_path = fs::path(output_path) / object_class.path().filename();

            float object_class_iou = compute_intersection_over_union(ground_truth_path.string(), prediction_path.string());
            object_classes_iou.push_back(object_class_iou);
        }
    }
   
    if (object_classes_iou.empty()) return 0.0f;
    return std::accumulate(object_classes_iou.begin(), object_classes_iou.end(), 0.0f) / static_cast<float>(object_classes_iou.size());
}

// Computes the average IoU between matched predicted and ground truth boxes
float compute_intersection_over_union(const std::string& ground_truth_path, const std::string& prediction_path) {
    std::map<std::string, std::map<std::string, std::vector<int>>> ground_truth_boxes = read_boxes_coordinates(ground_truth_path);
    std::map<std::string, std::map<std::string, std::vector<int>>> predicted_boxes = read_boxes_coordinates(prediction_path);

    float total_iou = 0.0f;
    int count = 0;

    for (const std::pair<const std::string, std::map<std::string, std::vector<int>>>& pair : ground_truth_boxes) {
        const std::string& file_id = pair.first;
        const std::map<std::string, std::vector<int>>& object_boxes = pair.second;

		for (const std::pair<const std::string, std::vector<int>>& object_box : object_boxes) {
            const std::string& object_id = object_box.first;
            float iou = compute_iou_if_present(object_id, object_box.second, predicted_boxes[file_id]);
			++count;
            if (iou > 0.0f) {
            	total_iou += iou;
        	} else {
            	std::cout << "No prediction for: " << object_id << std::endl;
        	}
        }
    }

    return count > 0 ? total_iou / static_cast<float>(count) : 0.0f;
}

// Reads bounding box coordinates from a directory of text files into a map
std::map<std::string, std::map<std::string, std::vector<int>>> read_boxes_coordinates(const std::string& directory_path) {
    std::map<std::string, std::map<std::string, std::vector<int>>> boxes;

    for (const fs::directory_entry& file_path : fs::directory_iterator(directory_path)) {
        std::ifstream file(file_path.path());
        // Find the position of the first '-'
        size_t dash_pos = file_path.path().filename().string().find('-');
        // Extract substring from the beginning up to the dash (not including it)
        std::string file_id = file_path.path().filename().string().substr(0, dash_pos);
        std::string object_id;
		int x_min, y_min, x_max, y_max;

        while (file >> object_id >> x_min >> y_min >> x_max >> y_max) {
            boxes[file_id][object_id] = {x_min, y_min, x_max, y_max};
        }
    }

    return boxes;
}

// Calculates IoU for a given pair of ground truth and predicted boxes
float compute_iou_if_present(const std::string& object_id, const std::vector<int>& ground_truth_box,
                             const std::map<std::string, std::vector<int>>& predicted_boxes) {
    if (predicted_boxes.find(object_id) != predicted_boxes.end()) {
        float iou = calculate_intersection_and_union_areas(ground_truth_box, predicted_boxes.at(object_id));
		return iou;
    }

    return 0.0f;
}

// Calculates the intersection and union areas of two bounding boxes
float calculate_intersection_and_union_areas(const std::vector<int>& first_box, const std::vector<int>& second_box) {
    int x_min = std::max(first_box[0], second_box[0]);
    int y_min = std::max(first_box[1], second_box[1]);
    int x_max = std::min(first_box[2], second_box[2]);
    int y_max = std::min(first_box[3], second_box[3]);

    int intersection_width = std::max(0, x_max - x_min);
    int intersection_height = std::max(0, y_max - y_min);
    int intersection_area = intersection_width * intersection_height;

    int first_box_area = (first_box[2] - first_box[0]) * (first_box[3] - first_box[1]);
    int second_box_area = (second_box[2] - second_box[0]) * (second_box[3] - second_box[1]);

    int union_area = first_box_area + second_box_area - intersection_area;

    return static_cast<float>(intersection_area) / static_cast<float>(union_area);
}

std::map<std::string, float> compute_detection_accuracy(const std::string& dataset_path, const std::string& output_path,
                                                     const std::string& ground_truths_path) {
    std::map<std::string, int> total_objects_by_class;
    std::map<std::string, int> true_positives_by_class;

    // Iterate over each object category (sugar box, mustard bottle, etc.)
    for (const fs::directory_entry& object_class : fs::directory_iterator(dataset_path)) {
        if (object_class.is_directory()) {
            std::string object_class_name = object_class.path().filename().string();  // Get the category name
            fs::path ground_truth_path = object_class.path() / ground_truths_path;
            fs::path prediction_path = fs::path(output_path) / object_class.path().filename();

            // Read bounding boxes for ground truths and predictions
            std::map<std::string, std::map<std::string, std::vector<int>>> ground_truth_boxes = read_boxes_coordinates(ground_truth_path.string());
            std::map<std::string, std::map<std::string, std::vector<int>>> predicted_boxes = read_boxes_coordinates(prediction_path.string());

            // Iterate through the ground truth boxes and compare with the predicted boxes
            for (const auto& pair : ground_truth_boxes) {
                const std::string& file_id = pair.first;
                const std::map<std::string, std::vector<int>>& object_boxes = pair.second;

                // Debug: Check the number of objects in this file
                std::cout << "In file " << file_id << ", found " << object_boxes.size() << " ground truth objects." << std::endl;

                for (const auto& object_box : object_boxes) {
                    const std::string& object_id = object_box.first;
                    std::string class_name = object_id.substr(0, object_id.find('_'));  // Get the class from the object ID (e.g., "004" from "004_sugar_box")

                    // Increment total object count for the object class
                    ++total_objects_by_class[class_name];

                    // Debug: Print the object ID being processed
                    std::cout << "Processing object " << object_id << " in class " << class_name << std::endl;

                    // Check if there is a predicted box for the same file and object ID
                    if (predicted_boxes.find(file_id) != predicted_boxes.end() &&
                        predicted_boxes[file_id].find(object_id) != predicted_boxes[file_id].end()) {
                        // Get the predicted box for the current object
                        float iou = compute_iou_if_present(object_id, object_box.second, predicted_boxes[file_id]);

                        // If IoU >= 0.5, consider it a true positive for this class
                        if (iou >= 0.5f) {
                            ++true_positives_by_class[class_name];
                        } else {
                            std::cout << "Object " << object_id << " in file " << file_id << " isn't a true positive" << std::endl;
                        }
                    } else {
                        std::cout << "No prediction found for object " << object_id << " in file " << file_id << std::endl;
                    }
                }
            }
        }
    }

    // Debug output: print True Positives and Total Objects for each class
    for (const auto& class_entry : total_objects_by_class) {
        const std::string& class_name = class_entry.first;
        std::cout << "Class: " << class_name << std::endl;
        std::cout << "True Positives: " << true_positives_by_class[class_name] << std::endl;
        std::cout << "Total Objects: " << class_entry.second << std::endl;
    }

    // Return accuracy by class
    std::map<std::string, float> accuracy_by_class;
    for (const auto& class_entry : total_objects_by_class) {
        const std::string& class_name = class_entry.first;
        int total_objects = class_entry.second;
        int true_positives = true_positives_by_class[class_name];

        if (total_objects > 0) {
            accuracy_by_class[class_name] = static_cast<float>(true_positives) / static_cast<float>(total_objects);
        } else {
            accuracy_by_class[class_name] = 0.0f;
        }
    }

    return accuracy_by_class;
}
