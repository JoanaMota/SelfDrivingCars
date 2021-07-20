import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt
from m2bk import *
from utils import plot_image
from feature_extraction import extract_features_harris, extract_features_sift, visualize_features, extract_features_fast, extract_features_orb, extract_features_dataset, print_info
from feature_matching import match_features, visualize_matches, match_features_knn, visualize_matches_knn, match_features_dataset
from motion_estimation import estimate_motion, estimate_trajectory
from m2bk import visualize_camera_movement

np.random.seed(1)
np.set_printoptions(threshold=sys.maxsize)

# Loading and Visualizing Data
nr_frames = 2
dataset_handler = DatasetHandler(nr_frames)

image_nr = 0
assert(image_nr <= nr_frames)

image = dataset_handler.images[image_nr]
# plot_image(image, "Image", "gray")

image_rgb = dataset_handler.images_rgb[image_nr]
# plot_image(image_rgb, "RGB Image")

depth = dataset_handler.depth_maps[image_nr]
# plot_image(depth, "Depth map", "jet")

print("Depth map shape: {0}".format(depth.shape))

v, u = depth.shape
depth_val = depth[v-1, u-1]
print("Depth value of the very bottom-right pixel of depth map {0} is {1:0.3f}"
      .format(image_nr, depth_val))
print("Calibration Matrix K:")
print(dataset_handler.k)
print("Number of frames in the dataset:")
print(dataset_handler.num_frames)

# -- Feature Extraction --
kp, des = extract_features_harris(image)
# SIFT
print("SIFT")
kp, des = extract_features_sift(image)
print_info(image_nr, kp)
visualize_features(image, kp, "Image with features from SIFT")
# FAST
print("FAST")
kp, des = extract_features_fast(image)
print_info(image_nr, kp)
visualize_features(image, kp, "Image with features from FAST and BRIEF")
# ORB
print("ORB")
kp, des = extract_features_orb(image)
print_info(image_nr, kp)
visualize_features(image, kp, "Image with features from ORB")

# Extract Features from all images
kp_list, des_list = extract_features_dataset(
    dataset_handler.images, extract_features_orb)
print("ORB")
image_nr = 0
for image in dataset_handler.images:
    print_info(image_nr, kp_list[image_nr])
    image_nr = +1

# Remember that the length of the returned by dataset_handler lists should be the same as the length of the image array
print("Length of images array: {0}".format(len(dataset_handler.images)))


# -- Feature Matching --
image_nr = 0
des1 = des_list[image_nr]
des2 = des_list[image_nr+1]

matches = match_features(des1, des2, "ORB")
print("Number of features matched in frames {0} and {1}: {2}".format(
    image_nr, image_nr+1, len(matches)))
visualize_matches(dataset_handler.images[image_nr], kp_list[image_nr],
                  dataset_handler.images[image_nr+1], kp_list[image_nr+1], matches, 5)


best_matches = match_features_knn(des1, des2, "ORB", 2, 0.5)
print("Number of features best matched in frames {0} and {1}: {2}".format(
    image_nr, image_nr+1, len(best_matches)))
visualize_matches_knn(dataset_handler.images[image_nr], kp_list[image_nr],
                      dataset_handler.images[image_nr+1], kp_list[image_nr+1], best_matches)

all_images_matches = match_features_dataset(des_list, match_features)

# -- Trajectory Estimation --

match = all_images_matches[image_nr]
kp1 = kp_list[image_nr]
kp2 = kp_list[image_nr+1]
k = dataset_handler.k
depth = dataset_handler.depth_maps[image_nr]

rmat, tvec, image1_points, image2_points = estimate_motion(
    match, kp1, kp2, k, depth1=depth)

print("Estimated rotation:\n {0}".format(rmat))
print("Estimated translation:\n {0}".format(tvec))

image1 = dataset_handler.images_rgb[image_nr]
image2 = dataset_handler.images_rgb[image_nr + 1]

image_move = visualize_camera_movement(
    image1, image1_points, image2, image2_points)

plot_image(image_move, "Camera Movement Visualization")

depth_maps = dataset_handler.depth_maps
trajectory = estimate_trajectory(
    estimate_motion, all_images_matches, kp_list, k, depth_maps=depth_maps)

i = 1
print("Camera location in point {0} is: \n {1}\n".format(
    i, trajectory[:, [i]]))

# Remember that the length of the returned by trajectory should be the same as the length of the image array
print("Length of trajectory: {0}".format(trajectory.shape[1]))

plt.show()
