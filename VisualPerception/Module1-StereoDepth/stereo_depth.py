import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import patches

import files_management


def compute_left_disparity_map(f_img_left, f_img_right):
    # Convert RGB image to Gray
    img_left_gray = cv2.cvtColor(f_img_left, cv2.COLOR_BGR2GRAY)
    img_right_gray = cv2.cvtColor(f_img_right, cv2.COLOR_BGR2GRAY)

    # Stereo BM
    stereo_BM = cv2.StereoBM_create(numDisparities=16*6, blockSize=11)
    # Stereo SGBM
    stereo_SGBM = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16*6,
        blockSize=11,
        P1=8 * 3 * 6 ** 2,
        P2=32 * 3 * 6 ** 2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    disp_left = stereo_SGBM.compute(
        img_left_gray, img_right_gray).astype(np.float32)/16
    return disp_left


def decompose_projection_matrix(f_p):
    k, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(f_p)
    t = t / t[3]
    return k, r, t


def calc_depth_map(f_disp, f_k, f_t_left, f_t_right):
    # Get focal length
    f = f_k[0][0]
    # Get baseline which is the distance between the left and right camera on the X axis
    b = f_t_left[1] - f_t_right[1]

    # Replace all 0 and -1 disparity with a small value
    # This way we avoid divisions by 0
    f_disp[f_disp == 0] = 0.1
    f_disp[f_disp == -1] = 0.1

    # Initialize the depth map to match the size of the disparity map
    depth_map = np.ones(f_disp.shape, np.single)
    # Compute the depth map = f*b/d
    depth_map = (f * b) / f_disp
    return depth_map


def locate_obstacle_in_image(f_image, f_obstacle_image):
    # Get the cross correlation heat map
    # Select te Template method option
    cross_correlation_map = cv2.matchTemplate(
        f_image, f_obstacle_image, cv2.TM_CCOEFF)
    # Get the position of the obstacle
    _, _, _, obstacle_location = cv2.minMaxLoc(cross_correlation_map)
    return cross_correlation_map, obstacle_location


def calculate_nearest_point(depth_map, obstacle_location, obstacle_img):

    # obstacle_width = obstacle_img.shape[1]
    # print(obstacle_width)
    # obstacle_height = obstacle_img.shape[0]
    # print(obstacle_height)
    # obstacle_min_x_pos = obstacle_location[0]
    # obstacle_max_x_pos = obstacle_location[0] + obstacle_width
    # obstacle_min_y_pos = obstacle_location[1]
    # obstacle_max_y_pos = obstacle_location[1] + obstacle_height
    obstacle_width = obstacle_img.shape[0]
    print(obstacle_width)
    obstacle_height = obstacle_img.shape[1]
    print(obstacle_height)
    obstacle_min_x_pos = obstacle_location[1]
    obstacle_max_x_pos = obstacle_location[1] + obstacle_width
    obstacle_min_y_pos = obstacle_location[0]
    obstacle_max_y_pos = obstacle_location[0] + obstacle_height
    print(obstacle_location)

    # Get the depth of the pixels within the bounds of the obstacle image, find the closest point in this rectangle
    obstacle_depth = depth_map[obstacle_min_x_pos:obstacle_max_x_pos,
                               obstacle_min_y_pos:obstacle_max_y_pos]
    print(obstacle_depth)
    closest_point_depth = obstacle_depth.min()
    print(closest_point_depth)

    # Create the obstacle bounding box
    obstacle_bbox = patches.Rectangle((obstacle_min_y_pos, obstacle_min_x_pos), obstacle_height, obstacle_width,
                                      linewidth=1, edgecolor='r', facecolor='none')

    return closest_point_depth, obstacle_bbox


# ------ Set-up ------
# Read the stereo-pair of images
img_left = files_management.read_left_image()
img_right = files_management.read_right_image()

# Use matplotlib to display the two images
_, image_cells = plt.subplots(1, 2, figsize=(20, 20))
image_cells[0].imshow(img_left)
image_cells[0].set_title('left image')
image_cells[1].imshow(img_right)
image_cells[1].set_title('right image')
plt.show(block=False)

# Read the calibration
p_left, p_right = files_management.get_projection_matrices()

# Use regular numpy notation instead of scientific one
np.set_printoptions(suppress=True)

print("p_left \n", p_left)
print("\np_right \n", p_right)

# ------ Estimating Depth ------
# Compute the disparity map using the fuction above
disp_left = compute_left_disparity_map(img_left, img_right)
# Show the left disparity map
plt.figure(figsize=(10, 10))
plt.imshow(disp_left)
plt.show(block=False)

# Decompose each matrix
k_left, r_left, t_left = decompose_projection_matrix(p_left)
k_right, r_right, t_right = decompose_projection_matrix(p_right)

# Display the matrices
print("k_left \n", k_left)
print("\nr_left \n", r_left)
print("\nt_left \n", t_left)
print("\nk_right \n", k_right)
print("\nr_right \n", r_right)
print("\nt_right \n", t_right)

# Get the depth map by calling the above function
depth_map_left = calc_depth_map(disp_left, k_left, t_left, t_right)

# Display the depth map
plt.figure(figsize=(8, 8), dpi=100)
plt.imshow(depth_map_left, cmap='flag')
plt.show(block=False)

# ------ Finding the distance to collision ------

# Get the image of the obstacle which in this case is already known
obstacle_image = files_management.get_obstacle_image()

# Show the obstacle image
plt.figure(figsize=(4, 4))
plt.imshow(obstacle_image)
plt.show(block=False)

# Gather the cross correlation map and the obstacle location in the image
cross_corr_map, obstacle_location = locate_obstacle_in_image(
    img_left, obstacle_image)

# Display the cross correlation heatmap
plt.figure(figsize=(10, 10))
plt.imshow(cross_corr_map)
plt.show(block=False)

# Print the obstacle location
print("obstacle_location \n", obstacle_location)

# Use the developed nearest point function to get the closest point depth and obstacle bounding box
closest_point_depth, obstacle_bbox = calculate_nearest_point(
    depth_map_left, obstacle_location, obstacle_image)

# Display the image with the bounding box displayed
fig, ax = plt.subplots(1, figsize=(10, 10))
ax.imshow(img_left)
ax.add_patch(obstacle_bbox)
plt.show()

# Print the depth of the nearest point
print("closest_point_depth {0:0.3f}".format(closest_point_depth))


# Printing results:
# Part 1. Read Input Data
img_left = files_management.read_left_image()
img_right = files_management.read_right_image()
p_left, p_right = files_management.get_projection_matrices()


# Part 2. Estimating Depth
disp_left = compute_left_disparity_map(img_left, img_right)
k_left, r_left, t_left = decompose_projection_matrix(p_left)
k_right, r_right, t_right = decompose_projection_matrix(p_right)
depth_map_left = calc_depth_map(disp_left, k_left, t_left, t_right)


# Part 3. Finding the distance to collision
obstacle_image = files_management.get_obstacle_image()
cross_corr_map, obstacle_location = locate_obstacle_in_image(
    img_left, obstacle_image)
closest_point_depth, obstacle_bbox = calculate_nearest_point(
    depth_map_left, obstacle_location, obstacle_image)


# Print Result Output
print("Left Projection Matrix Decomposition:\n {0}".format([k_left.tolist(),
                                                            r_left.tolist(),
                                                            t_left.tolist()]))
print("\nRight Projection Matrix Decomposition:\n {0}".format([k_right.tolist(),
                                                               r_right.tolist(),
                                                               t_right.tolist()]))
print(
    "\nObstacle Location (left-top corner coordinates):\n {0}".format(list(obstacle_location)))
print("\nClosest point depth (meters):\n {0}".format(closest_point_depth))
