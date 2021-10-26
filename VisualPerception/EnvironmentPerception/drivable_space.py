import numpy as np
from m6bk import compute_plane, dist_to_plane


def xy_from_depth(depth, k):
    """
    Computes the x, and y coordinates of every pixel in the image using the depth map and the calibration matrix.

    Arguments:
    depth -- tensor of dimension (H, W), contains a depth value (in meters) for every pixel in the image.
    k -- tensor of dimension (3x3), the intrinsic camera matrix

    Returns:
    x -- tensor of dimension (H, W) containing the x coordinates of every pixel in the camera coordinate frame.
    y -- tensor of dimension (H, W) containing the y coordinates of every pixel in the camera coordinate frame.
    """
    # Get the shape of the depth tensor
    depth_shape = depth.shape

    # Grab required parameters from the K matrix
    f = k[0][0]
    cu = k[0][2]
    cv = k[1][2]

    # Generate a grid of coordinates corresponding to the shape of the depth map
    x = np.zeros(depth_shape)
    y = np.zeros(depth_shape)

    # Compute x and y coordinates
    for row in range(depth_shape[0]):
        for column in range(depth_shape[1]):
            x[row][column] = ((row-cu)*depth[row][column])/f
            y[row][column] = ((column-cv)*depth[row][column])/f

    return x, y


def ransac_plane_fit(xyz_data):
    """
    Computes plane coefficients a,b,c,d of the plane in the form ax+by+cz+d = 0
    using ransac for outlier rejection.

    Arguments:
    xyz_data -- tensor of dimension (3, N), contains all data points from which random sampling will proceed.
    num_itr --
    distance_threshold -- Distance threshold from plane for a point to be considered an inlier.

    Returns:
    p -- tensor of dimension (1, 4) containing the plane parameters a,b,c,d
    """

    # Set thresholds:
    num_itr = 1000  # RANSAC maximum number of iterations
    # RANSAC minimum number of inliers
    min_num_inliers = int(0.9*xyz_data.shape[1])
    # Maximum distance from point to plane for point to be considered inlier
    distance_threshold = 0.00001
    num_inliers_max = 0
    xyz_max = np.zeros((3, 4))
    min_num_rand_points = 3

    for i in range(num_itr):
        # Step 1: Choose a minimum of 3 points from xyz_data at random.
        rand_index = np.random.choice(
            a=xyz_data.shape[1], size=min_num_rand_points)
        xyz = xyz_data[:, rand_index]
        # Step 2: Compute plane model
        plane = compute_plane(xyz)

        # Step 3: Find number of inliers
        xx = np.array(xyz_data[0][0:]).T
        yy = np.array(xyz_data[1][0:]).T
        zz = np.array(xyz_data[2][0:]).T
        distances = np.abs(dist_to_plane(plane, xx, yy, zz))
        num_inliers = np.sum(distances < distance_threshold)

        # Step 4: Check if the current number of inliers is greater than all previous iterations and keep the inlier set with the largest number of points.
        if num_inliers > num_inliers_max:
            xyz_max = xyz
            num_inliers_max = num_inliers
        # Step 5: Check if stopping criterion is satisfied and break.
        if num_inliers_max >= min_num_inliers:
            break

    # Step 6: Recompute the model parameters using largest inlier set.
    plane = compute_plane(xyz_max)

    return plane
