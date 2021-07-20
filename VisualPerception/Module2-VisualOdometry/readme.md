# Visual Odometry for Localization in Autonomous Driving

To achieve this we will:

- Extract features from the photographs taken with a camera setup on the vehicle.
- Use the extracted features to find matches between the features in different photographs.
- Use the found matches to estimate the camera motion between subsequent photographs.
- Use the estimated camera motion to build the vehicle trajectory.

## 1 - Feature Extraction

### 1.1 - Feature Detection

**Feature:** Point of interest (keypoint) in an image defined by its image pixel coordinates, scale, orientation and some other parameters.

**Feature Extractor Algorithms:**

- [Harris](https://docs.opencv.org/3.4.3/dc/d0d/tutorial_py_features_harris.html)
- [FAST](https://docs.opencv.org/3.4.3/df/d0c/tutorial_py_fast.html)
- [SURF](https://docs.opencv.org/3.4.3/df/dd2/tutorial_py_surf_intro.html)
- [SIFT](https://docs.opencv.org/3.4.3/da/df5/tutorial_py_sift_intro.html)
- [ORB](https://docs.opencv.org/3.4.3/d1/d89/tutorial_py_orb.html)

To get the descriptor from the Feature Extractor Algorithms that only provide the keypoints you will need to use a separate instance of **Descriptor Extractor**.

Some Feature Extractor Algorithms already work as Descriptor Extractors.

### 1.2 - Descriptor Extraction

**Descriptor:** An N-dimensional vector that provides a summary of the image information around the detected feature. It must be:

- Robust and invariant to translations, rotations, scales and illumination changes;
- Distinctive from all others;
- Compact and efficient to enable lower computational times.

**Descriptor Extractor Algorithms:**

- [BRIEF](https://docs.opencv.org/3.4.3/dc/d7d/tutorial_py_brief.html)
- [SIFT](https://docs.opencv.org/3.4.3/da/df5/tutorial_py_sift_intro.html)
- [SURF](https://docs.opencv.org/3.4.3/df/dd2/tutorial_py_surf_intro.html)
- [ORB](https://docs.opencv.org/3.4.3/d1/d89/tutorial_py_orb.html)

| Algorithm | Keypoints | Descriptors |                                                          Velocity and Key factors                                                          |
| :-------: | :-------: | :---------: | :----------------------------------------------------------------------------------------------------------------------------------------: |
|  Harris   |    yes    |     no      |                                                                                                                                            |
|   SIFT    |    yes    |     yes     |                                                        Works also for scaled images                                                        |
|   SURF    |    yes    |     yes     |                                                         Similar to SIFT but faster                                                         |
|   FAST    |    yes    |     no      |                                                                                                                                            |
|   BRIEF   |    no     |     yes     |                                                      Binary string based descriptors.                                                      |
|   BRISK   |    yes    |     yes     |                                                      Binary string based descriptors.                                                      |
|    ORB    |    yes    |     yes     | Binary string based descriptors. An efficient alternative to SIFT or SURF but FREE. Fusion of FAST keypoint detector and BRIEF descriptor. |

[Checkout how to use some of the feature extraction algorithms here](https://github.com/JoanaMota/SelfDrivingCars/blob/main/VisualPerception/Module2-VisualOdometry/feature_extraction.py)

# 2 - [Feature Matching](https://docs.opencv.org/3.4.3/dc/dc3/tutorial_py_matcher.html)

Is basically to match features in one image with others.

**Examples of Matchers:**

- Brute-Force Matcher
- FLANN Matcher

## Brute-Force Matcher

Takes the descriptor of one feature in first set which is then matched with all other features in second set using some distance calculation. And the closest one is returned.

To create a BFMatcher object `cv.BFMatcher()` you need to define the **NormType** and bool variable **crossCheck**.

[**NormTypes**](https://docs.opencv.org/3.4.3/d2/de8/group__core__array.html#gad12cefbcb5291cf958a85b4b67b6149f) selection:

- `cv.NORM_L2` and `cv.NORM_L1` -> SIFT, SURF.
- `cv.NORM_HAMMING` -> ORB, BRIEF, BRISK

Use **CrossCheck** as `True` for better results. Don't know exactly what that means :confused:

Then you have 2 options:

- Use `match()` to get all the matches, sort them in ascending order of their distances so that best matches (with low distance) come to front. The lower, the better it is. :grin:
- Used `knnMatch()` to get k best matches and then apply ratio test.

## FLANN - Fast Library for Approximate Nearest Neighbors

It works faster than BFMatcher for large datasets.

[Checkout how to use some of the feature matching algorithms here](https://github.com/JoanaMota/SelfDrivingCars/blob/main/VisualPerception/Module2-VisualOdometry/feature_matching.py)

# 3 - Trajectory Estimation

So now we reach our end goal, determine the pose of our car, so its Rotation and Translation `[R|t]` matrix. This is called the [Perspective-n-Point](https://en.wikipedia.org/wiki/Perspective-n-Point) problem. Fortunately, OpenCV has a robust implementation of algorithm to solve the PnP in `cv2.solvePnP()` and `cv2.solvePnPRansac()`. These functions take three arguments.

**Input:**

- _objectPoints_ - a numpy array of features in 3D world coordinates for `frame[k-1]`
- _magePoints_ - a numpy array of corresponding image points in 2D coordinates for `frame[k]`
- _cameraMatrix_ - Intrinsic calibration matrix `K`

Transform pixel coordinates to camera coordinates

> [su;sv;s] = K [R|t] [x;y;z;1]

[Checkout how to estimate the camera trajectory here](https://github.com/JoanaMota/SelfDrivingCars/blob/main/VisualPerception/Module2-VisualOdometry/motion_estimation.py)
