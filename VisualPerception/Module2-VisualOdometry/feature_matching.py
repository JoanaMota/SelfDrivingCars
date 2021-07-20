import numpy as np
import cv2
from utils import plot_image


def match_features(des1, des2, algorithm):
    """
    Match features from two images

    Arguments:
    des1 -- list of the keypoint descriptors in the first image
    des2 -- list of the keypoint descriptors in the second image
    algorithm -- feature extraction algorithm used

    Returns:
    match -- ordered list of matched features from two images. Each match[i] is k or less matches for the same query descriptor
    """
    if "ORB" == algorithm:
        normType = cv2.NORM_HAMMING
    else:
        print("Invalid Feature Extraction Algorithm")
        return 0
    # create BFMatcher object
    bf = cv2.BFMatcher(normType, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    return matches


def match_features_knn(des1, des2, algorithm, k, ratio):
    """
    The best k match features from two images

    Arguments:
    des1 -- list of the keypoint descriptors in the first image
    des2 -- list of the keypoint descriptors in the second image
    algorithm -- feature extraction algorithm used
    k -- Count of best matches found per each query descriptor or less if a query descriptor has less than k possible matches in total
    ratio -- ratio for distance limit

    Returns:
    good -- list of best matched features from two images.
    """
    if "ORB" == algorithm:
        normType = cv2.NORM_HAMMING
    else:
        print("Invalid Feature Extraction Algorithm")
        return 0
    # create BFMatcher object
    bf = cv2.BFMatcher(normType)
    # Match descriptors.
    matches = bf.knnMatch(des1, des2, k)
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < ratio*n.distance:
            good.append([m])

    return good


def visualize_matches(img1, kp1, img2, kp2, matches, nr=0):
    """
    Visualize extracted features in the image

    Arguments:
    image -- a grayscale image
    kp -- list of the extracted keypoints
    name -- algo name used to extract the features
    nr -- select number of matches

    Returns:
    """
    if nr == 0:
        outImg = cv2.drawMatches(img1, kp1, img2, kp2,
                                 matches, None, flags=2)
        plot_image(outImg, "Feature Matches")
    else:
        outImg = cv2.drawMatches(img1, kp1, img2, kp2,
                                 matches[:nr], None, flags=2)
        plot_image(outImg, "Best {} Feature Matches".format(nr))


def visualize_matches_knn(img1, kp1, img2, kp2, matches):
    """
    Visualize extracted features in the image

    Arguments:
    image -- a grayscale image
    kp -- list of the extracted keypoints
    name -- algo name used to extract the features

    Returns:
    """
    outImg = cv2.drawMatchesKnn(img1, kp1, img2, kp2,
                                matches, None, flags=2)
    plot_image(outImg, "Best MatchesKnn")


def match_features_dataset(des_list, match_features_func):
    """
    Match features for each subsequent image pair in the dataset

    Arguments:
    des_list -- a list of descriptors for each image in the dataset
    match_features_func -- a function which maches features between a pair of images

    Returns:
    matches -- list of matches for each subsequent image pair in the dataset.
               Each matches[i] is a list of matched features from images i and i + 1

    """
    matches = []
    algorithm = "ORB"
    i = 0
    while i < len(des_list)-1:
        matches.append(match_features_func(
            des_list[i], des_list[i+1], algorithm))
        i = +1
    return matches


def match_features_knn_dataset(des_list, match_features_func, k, ratio):
    """
    Match features for each subsequent image pair in the dataset

    Arguments:
    des_list -- a list of descriptors for each image in the dataset
    match_features_func -- a function which maches features between a pair of images

    Returns:
    matches -- list of matches for each subsequent image pair in the dataset.
               Each matches[i] is a list of matched features from images i and i + 1

    """
    matches = []
    algorithm = "ORB"
    i = 0
    while i+1 < len(des_list):
        matches.append(match_features_func(
            des_list[i], des_list[i+1], algorithm, k, ratio))
        i = +1
    return matches


def filter_matches_distance(match, dist_threshold):
    """
    Filter matched features from two images by distance between the best matches

    Arguments:
    match -- list of matched features from two images
    dist_threshold -- maximum allowed relative distance between the best matches, (0.0, 1.0) 

    Returns:
    filtered_match -- list of good matches, satisfying the distance threshold
    """
    filtered_match = []
    for it in range(len(match)-1):
        if match[it].distance < match[it+1].distance:
            filtered_match.append(match[it])

    return filtered_match


def filter_matches_dataset(filter_matches_distance, matches, dist_threshold):
    """
    Filter matched features by distance for each subsequent image pair in the dataset

    Arguments:
    filter_matches_distance -- a function which filters matched features from two images by distance between the best matches
    matches -- list of matches for each subsequent image pair in the dataset. 
               Each matches[i] is a list of matched features from images i and i + 1
    dist_threshold -- maximum allowed relative distance between the best matches, (0.0, 1.0) 

    Returns:
    filtered_matches -- list of good matches for each subsequent image pair in the dataset. 
                        Each matches[i] is a list of good matches, satisfying the distance threshold

    """
    filtered_matches = []

    for match in matches:
        filtered_match = filter_matches_distance(match, dist_threshold)
        filtered_matches.append(filtered_match)

    return filtered_matches
