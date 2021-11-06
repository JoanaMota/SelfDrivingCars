import numpy as np
import cv2
from utils import plot_image


def filter_detections_by_segmentation(detections, segmentation_output):
    """
    Filter 2D detection output based on a semantic segmentation map.

    Arguments:
    detections -- tensor of dimension (N, 5) containing detections in the form of [Class, x_min, y_min, x_max, y_max, score].

    segmentation_output -- tensor of dimension (HxW) containing pixel category labels.

    Returns:
    filtered_detections -- tensor of dimension (N, 5) containing detections in the form of [Class, x_min, y_min, x_max, y_max, score].

    """
    # Set ratio threshold:
    # If 1/3 of the total pixels belong to the target category, the detection is correct.
    ratio_threshold = 0.3

    filtered_detections = []
    for detection in detections:

        # Step 1: Compute number of pixels belonging to the category for every detection.
        bounding_box = segmentation_output[int(float(detection[2])):int(float(detection[4])),
                                           int(float(detection[1])):int(float(detection[3]))]
        nr_pixels = np.sum(bounding_box == 10)

        # Step 2: Divide the computed number of pixels by the area of the bounding box (total number of pixels).
        x = float(detection[3]) - float(detection[1])
        y = float(detection[4]) - float(detection[2])
        bb_area = x * y

        ratio = nr_pixels / bb_area
        print(ratio)

        # Step 3: If the ratio is greater than a threshold keep the detection. Else, remove the detection from the list of detections.
        if ratio >= ratio_threshold:
            filtered_detections.append(detection)

    return filtered_detections


def find_min_distance_to_detection(detections, x, y, z):
    """
    Filter 2D detection output based on a semantic segmentation map.

    Arguments:
    detections -- tensor of dimension (N, 5) containing detections in the form of [Class, x_min, y_min, x_max, y_max, score].

    x -- tensor of dimension (H, W) containing the x coordinates of every pixel in the camera coordinate frame.
    y -- tensor of dimension (H, W) containing the y coordinates of every pixel in the camera coordinate frame.
    z -- tensor of dimensions (H,W) containing the z coordinates of every pixel in the camera coordinate frame.
    Returns:
    min_distances -- tensor of dimension (N, 1) containing distance to impact with every object in the scene.

    """
    min_distances = []
    for detection in detections:
        # Step 1: Compute distance of every pixel in the detection bounds
        x_bb = x[int(float(detection[2])):int(float(detection[4])),
                 int(float(detection[1])):int(float(detection[3]))]
        y_bb = y[int(float(detection[2])):int(float(detection[4])),
                 int(float(detection[1])):int(float(detection[3]))]
        z_bb = z[int(float(detection[2])):int(float(detection[4])),
                 int(float(detection[1])):int(float(detection[3]))]
        x_sqr = np.multiply(x_bb, x_bb)
        y_sqr = np.multiply(y_bb, y_bb)
        z_sqr = np.multiply(z_bb, z_bb)
        detection_distance = np.sqrt(x_sqr + y_sqr + z_sqr)
        # Step 2: Find minimum distance
        min_distances.append(np.min(detection_distance))
    return min_distances
