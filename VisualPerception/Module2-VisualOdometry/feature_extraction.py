import numpy as np
import cv2
from utils import plot_image


def extract_features_harris(image):
    """
    Find keypoints and descriptors for the image

    Arguments:
    image -- a grayscale image

    Returns:
    kp -- list of the extracted keypoints (features) in an image
    des -- list of the keypoint descriptors in an image
    """
    gray = np.float32(image)
    # dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    # dst = cv2.dilate(dst, None)
    # plt.imshow(dst)
    corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
    corners = np.int0(corners)
    for i in corners:
        x, y = i.ravel()
        cv2.circle(image, (x, y), 3, 255, -1)
    # plot_image(image, "Grayscale Image with Harris corners")
    kp = 0
    des = 0
    return kp, des


def extract_features_sift(image):
    """
    Find keypoints and descriptors for the image
    using the SIFT algorithm

    Arguments:
    image -- a grayscale image

    Returns:
    kp -- list of the extracted keypoints (features) in an image
    des -- list of the keypoint descriptors in an image
    """
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(image, None)
    return kp, des


def extract_features_fast(image):
    """
    Find keypoints and descriptors for the image
    using the FAST feature extractor and BRIEF 
    descriptor extractor

    Arguments:
    image -- a grayscale image

    Returns:
    kp -- list of the extracted keypoints (features) in an image
    des -- list of the keypoint descriptors in an image
    """
    # FAST is only a keypoint detector
    # you can't get any feature descriptors from it
    fast = cv2.FastFeatureDetector_create()
    kp = fast.detect(image, None)
    # we need a descriptor extractor
    # lest use BRIEF
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    kp, des = brief.compute(image, kp)
    return kp, des


def extract_features_orb(image):
    """
    Find keypoints and descriptors for the image
    using the ORB feature extractor

    Arguments:
    image -- a grayscale image

    Returns:
    kp -- list of the extracted keypoints (features) in an image
    des -- list of the keypoint descriptors in an image
    """
    # Initiate ORB detector
    orb = cv2.ORB_create()
    # find the keypoints with ORB
    kp = orb.detect(image, None)
    # compute the descriptors with ORB
    kp, des = orb.compute(image, kp)
    return kp, des


def extract_features_dataset(images, extract_features_function):
    """
    Find keypoints and descriptors for each image in the dataset

    Arguments:
    images -- a list of grayscale images
    extract_features_function -- a function which finds features (keypoints and descriptors) for an image

    Returns:
    kp_list -- a list of keypoints for each image in images
    des_list -- a list of descriptors for each image in images

    """
    kp_list = []
    des_list = []
    for image in images:
        kp, des = extract_features_function(image)
        kp_list.append(kp)
        des_list.append(des)

    return kp_list, des_list


def print_info(image_nr, kp):
    print("Number of features detected in frame {0}: {1}\n".format(
        image_nr, len(kp)))
    print("Coordinates of the first keypoint in frame {0}: {1}".format(
        image_nr, str(kp[0].pt)))


def visualize_features(image, kp, algo_name):
    """
    Visualize extracted features in the image

    Arguments:
    image -- a grayscale image
    kp -- list of the extracted keypoints
    name -- algo name used to extract the features

    Returns:
    """
    display = cv2.drawKeypoints(image, kp, None)
    plot_image(display, algo_name)
