import numpy as np
import cv2


def estimate_motion(match, kp1, kp2, k, depth1=None):
    """
    Estimate camera motion from a pair of subsequent image frames

    Arguments:
    match -- list of matched features from the pair of images
    kp1 -- list of the keypoints in the first image
    kp2 -- list of the keypoints in the second image
    k -- camera calibration matrix 

    Optional arguments:
    depth1 -- a depth map of the first frame. This argument is not needed if you use Essential Matrix Decomposition

    Returns:
    rmat -- recovered 3x3 rotation numpy matrix
    tvec -- recovered 3x1 translation numpy vector
    image1_points -- a list of selected match coordinates in the first image. image1_points[i] = [u, v], where u and v are 
                     coordinates of the i-th match in the image coordinate system
    image2_points -- a list of selected match coordinates in the second image. image1_points[i] = [u, v], where u and v are 
                     coordinates of the i-th match in the image coordinate system

    """
    rmat = np.eye(3)
    tvec = np.zeros((3, 1))
    image1_points = []
    image2_points = []
    objectpoints = []

    for it in range(len(match)-1):
        m = match[it]
        # get first img matched keypoints
        u1, v1 = kp1[m.queryIdx].pt

        # get second img matched keypoints
        u2, v2 = kp2[m.trainIdx].pt

        s = depth1[int(v1), int(u1)]
        if s < 1000:
            # Transform pixel coordinates to camera coordinates
            pixel_coord = np.linalg.inv(k) @ (s * np.array([u1, v1, 1]))
            image2_points.append([u2, v2])
            image1_points.append([u1, v1])
            objectpoints.append(pixel_coord)

    objectpoints = np.vstack(objectpoints)
    image_points = np.array(image2_points)
    _, rvec, tvec, _ = cv2.solvePnPRansac(objectpoints, image_points, k, None)

    rmat, _ = cv2.Rodrigues(rvec)

    return rmat, tvec, image1_points, image2_points


def estimate_trajectory(estimate_motion, matches, kp_list, k, depth_maps=[]):
    """
    Estimate complete camera trajectory from subsequent image pairs

    Arguments:
    estimate_motion -- a function which estimates camera motion from a pair of subsequent image frames
    matches -- list of matches for each subsequent image pair in the dataset. 
               Each matches[i] is a list of matched features from images i and i + 1
    des_list -- a list of keypoints for each image in the dataset
    k -- camera calibration matrix 

    Optional arguments:
    depth_maps -- a list of depth maps for each frame. This argument is not needed if you use Essential Matrix Decomposition

    Returns:
    trajectory -- a 3xlen numpy array of the camera locations, where len is the lenght of the list of images and   
                  trajectory[:, i] is a 3x1 numpy vector, such as:

                  trajectory[:, i][0] - is X coordinate of the i-th location
                  trajectory[:, i][1] - is Y coordinate of the i-th location
                  trajectory[:, i][2] - is Z coordinate of the i-th location

                  * Consider that the origin of your trajectory cordinate system is located at the camera position 
                  when the first image (the one with index 0) was taken. The first camera location (index = 0) is given 
                  at the initialization of this function

    """
    trajectory = np.zeros((3, len(matches) + 1))

    # Initialize camera pose
    camera_pose = np.eye(4)

    # Iterate through the matched features
    for i in range(len(matches)):
        # Estimate camera motion between a pair of images
        rmat, tvec, image1_points, image2_points = estimate_motion(
            matches[i], kp_list[i], kp_list[i + 1], k, depth_maps[i])

        # Determine current pose from rotation and translation matrices
        current_pose = np.eye(4)
        current_pose[0:3, 0:3] = rmat
        current_pose[0:3, 3] = tvec.T

        # Build the robot's pose from the initial position by multiplying previous and current poses
        camera_pose = camera_pose @ np.linalg.inv(current_pose)

        # Calculate current camera position from origin
        position = camera_pose @ np.array([0., 0., 0., 1.])

        # Build trajectory
        trajectory[:, i + 1] = position[0:3]
    return trajectory
