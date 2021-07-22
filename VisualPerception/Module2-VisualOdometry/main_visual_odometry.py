import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt
from m2bk import *
from utils import plot_image
from feature_extraction import extract_features_harris, extract_features_sift, visualize_features, extract_features_fast, extract_features_orb, extract_features_dataset, print_info
from feature_matching import match_features, visualize_matches, match_features_knn, visualize_matches_knn, match_features_dataset, match_features_knn_dataset, filter_matches_distance, filter_matches_dataset
from motion_estimation import estimate_motion, estimate_trajectory
from m2bk import visualize_camera_movement, visualize_trajectory

dataset_handler = DatasetHandler()
k = dataset_handler.k
images = dataset_handler.images

img_nr = 10
# Part 1. Features Extraction
print("Extracting Features")
kp_list, des_list = extract_features_dataset(images, extract_features_orb)
print("Number of features detected in frame {0}: {1}".format(
    img_nr, len(kp_list[img_nr])))
print("Coordinates of the first keypoint in frame {0}: {1}\n".format(
    img_nr, str(kp_list[img_nr][0].pt)))
print("Finished Extracting Features")

# Part II. Feature Matching
print("Feature Matching")
matches = match_features_dataset(des_list, match_features)
print(len(matches))
print("Finished Feature Matching")

# Set to True if you want to use filtered matches or False otherwise
is_main_filtered_m = True
if is_main_filtered_m:
    dist_threshold = 0.75
    filtered_matches = filter_matches_dataset(
        filter_matches_distance, matches, dist_threshold)
    matches = filtered_matches
print(len(matches))
print("Trajectory Estimation")
# Part III. Trajectory Estimation
depth_maps = dataset_handler.depth_maps
trajectory = estimate_trajectory(
    estimate_motion, matches, kp_list, k, depth_maps=depth_maps)


#!!! Make sure you don't modify the output in any way
# Print Submission Info
print("Trajectory X:\n {0}".format(trajectory[0, :].reshape((1, -1))))
print("Trajectory Y:\n {0}".format(trajectory[1, :].reshape((1, -1))))
print("Trajectory Z:\n {0}".format(trajectory[2, :].reshape((1, -1))))


visualize_trajectory(trajectory)
