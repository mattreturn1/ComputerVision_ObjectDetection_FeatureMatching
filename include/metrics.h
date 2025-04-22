// Michele Brigandì

#ifndef METRICS_H
#define METRICS_H

#include <string>
#include <vector>
#include <map>

float compute_mean_intersection_over_union(const std::string& dataset_path, const std::string& output_path,
                                           const std::string& ground_truths_path = "labels");

float compute_intersection_over_union(const std::string& ground_truth_path, const std::string& prediction_path);

std::map<std::string, std::map<std::string, std::vector<int>>> read_boxes_coordinates(const std::string& directory_path);

float compute_iou_if_present(const std::string& object_id, const std::vector<int>& ground_truth_box,
                             const std::map<std::string, std::vector<int>>& predicted_boxes);

float calculate_intersection_and_union_areas(const std::vector<int>& first_box, const std::vector<int>& second_box);

float compute_detection_accuracy(const std::string& dataset_path, const std::string& output_path,
                                 const std::string& ground_truths_path = "labels");

#endif // METRICS_H
