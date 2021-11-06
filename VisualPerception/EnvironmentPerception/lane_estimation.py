import numpy as np
import cv2
from utils import plot_image
from m6bk import get_slope_intecept


def estimate_lane_lines(segmentation_output):
    """
    Estimates lines belonging to lane boundaries. Multiple lines could correspond to a single lane.

    Arguments:
    segmentation_output -- tensor of dimension (H,W), containing semantic segmentation neural network output
    minLineLength -- Scalar, the minimum line length
    maxLineGap -- Scalar, dimension (Nx1), containing the z coordinates of the points

    Returns:
    lines -- tensor of dimension (N, 4) containing lines in the form of [x_1, y_1, x_2, y_2], where [x_1,y_1] and [x_2,y_2] are
    the coordinates of two points on the line in the (u,v) image coordinate frame.
    """
    # Step 1: Create an image with pixels belonging to lane boundary categories from the output of semantic segmentation
    lane_boundaries = np.zeros(segmentation_output.shape)
    lane_boundaries[segmentation_output == 6] = 1
    lane_boundaries[segmentation_output == 8] = 1

    # Step 2: Perform Edge Detection using cv2.Canny()
    lane_boundaries_u8 = np.uint8(lane_boundaries)
    plot_image(lane_boundaries_u8, "Lane Markings", 'gray')

    lane_edges = cv2.Canny(lane_boundaries_u8, 0, 0.3)
    plot_image(lane_boundaries_u8, "Lane Edges", 'gray')

    # Step 3: Perform Line estimation using cv2.HoughLinesP()
    lines = cv2.HoughLinesP(lane_edges, rho=1, theta=np.pi /
                            180, threshold=100, minLineLength=20, maxLineGap=65)
    lines = np.squeeze(lines)
    # Note: Make sure dimensions of returned lines is (N x 4)

    return lines


def merge_lane_lines(lines, min_y, max_y):
    """
    Merges lane lines to output a single line per lane, using the slope and intercept as similarity measures.
    Also, filters horizontal lane lines based on a minimum slope threshold.

    Arguments:
    lines -- tensor of dimension (N, 4) containing lines in the form of [x_1, y_1, x_2, y_2],
    the coordinates of two points on the line.

    Returns:
    merged_lines -- tensor of dimension (N, 4) containing lines in the form of [x_1, y_1, x_2, y_2],
    the coordinates of two points on the line.
    """

    # Step 0: Define thresholds
    slope_similarity_threshold = 0.1
    intercept_similarity_threshold = 40
    min_slope_threshold = 0.3

    # Step 1: Get slope and intercept of lines
    slopes, intercepts = get_slope_intecept(lines)
    print(slopes)
    print(intercepts)

    # Step 2: Determine lines with slope less than horizontal slope threshold.
    lines = lines[np.abs(slopes) > min_slope_threshold]
    intercepts = intercepts[np.abs(slopes) > min_slope_threshold]
    slopes = slopes[np.abs(slopes) > min_slope_threshold]
    assert len(slopes) == len(intercepts)
    print(slopes)
    print(intercepts)

    final_slopes = []
    final_intercepts = []
    slopes_temp = []
    intercepts_temp = []
    slopes_temp.append(slopes[0])
    intercepts_temp.append(intercepts[0])
    # Step 3: Iterate over all remaining slopes and intercepts and cluster lines that are close to each other using a slope and intercept threshold.
    for slope, intercept in zip(slopes, intercepts):
        for i in range(len(slopes_temp)):
            if np.absolute(slope - slopes_temp[i]) >= slope_similarity_threshold and np.absolute(intercept - intercepts_temp[i]) >= intercept_similarity_threshold:
                final_slopes.append(slope)
                final_intercepts.append(intercept)
            else:
                slopes_temp.append(slope)
                intercepts_temp.append(slope)

    # Step 4: Merge all lines in clusters using mean averaging
    final_slopes.append(slopes_temp[0])
    final_intercepts.append(intercepts_temp[0])
    new_lines = []

    for slope, intercept, in zip(final_slopes, final_intercepts):
        x1 = (min_y - intercept) / slope
        x2 = (max_y - intercept) / slope
        new_lines.append([x1, min_y, x2, max_y])

    # Note: Make sure dimensions of returned lines is (N x 4)
    return np.array(new_lines)
