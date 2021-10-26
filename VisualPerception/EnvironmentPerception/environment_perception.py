import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt
from m6bk import *
from utils import plot_image
from drivable_space import xy_from_depth, ransac_plane_fit
from lane_estimation import estimate_lane_lines

np.random.seed(1)
np.set_printoptions(precision=2, threshold=sys.maxsize)

# ----- Step 0 : Loading Data

dataset_handler = DatasetHandler()

print("Current Image frame:" + str(dataset_handler.current_frame))

image = dataset_handler.image
plot_image(image, "Start Image")

k = dataset_handler.k
print("Camera Calibration Matrix: ")
print(k)

depth = dataset_handler.depth
plot_image(depth, "Depth Image", "jet")

segmentation = dataset_handler.segmentation
plot_image(segmentation, "Segmented Image")

colored_segmentation = dataset_handler.vis_segmentation(segmentation)
plot_image(colored_segmentation, "Colored Segmented Image")
dataset_handler.set_frame(0)

# ----- Step 1 : Drivable Space Estimation in 3D
# 1.1 - Estimating the x, y, and z coordinates of every pixel in the image

k = dataset_handler.k

z = dataset_handler.depth

x, y = xy_from_depth(z, k)

print('x[800,800] = ' + str(x[800, 800]))
print('y[800,800] = ' + str(y[800, 800]))
print('z[800,800] = ' + str(z[800, 800]) + '\n')

print('x[500,500] = ' + str(x[500, 500]))
print('y[500,500] = ' + str(y[500, 500]))
print('z[500,500] = ' + str(z[500, 500]) + '\n')

# 1.2 - Estimating The Ground Plane Using RANSAC
# Get road mask by choosing pixels in segmentation output with value 7
road_mask = np.zeros(segmentation.shape)
road_mask[segmentation == 7] = 1

# Show road mask
plot_image(road_mask, "Road Mask")

# Get x,y, and z coordinates of pixels in road mask
x_ground = x[road_mask == 1]
y_ground = y[road_mask == 1]
z_ground = dataset_handler.depth[road_mask == 1]
xyz_ground = np.stack((x_ground, y_ground, z_ground))

p_final = ransac_plane_fit(xyz_ground)
print('Ground Plane: ' + str(p_final))

dist = np.abs(dist_to_plane(p_final, x, y, z))

ground_mask = np.zeros(dist.shape)
ground_mask[dist < 0.1] = 1
ground_mask[dist > 0.1] = 0
plot_image(ground_mask, "Ground Plane Mask")

# dataset_handler.plot_free_space(ground_mask)

# ----- Step 2 : Lane Estimation Using The Semantic Segmentation Output
# 2.2 - Estimate Lane Boundary Proposals
lane_lines = estimate_lane_lines(segmentation)
print(lane_lines.shape)

plot_image(dataset_handler.vis_lanes(lane_lines), "Lane Markings")

plt.show()
