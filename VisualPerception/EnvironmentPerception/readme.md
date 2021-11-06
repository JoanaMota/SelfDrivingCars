# Environment Perception For Self-Driving Cars

How can we implement an environment perception stack for self-driving cars?

We will ...

Use the output from semantic segmentation neural networks to:

- implement drivable space estimation in 3D;
- implement lane estimation;
- filter out unreliable estimates in the output of 2D object detectors.

Use the filtered 2D object detection output to determine how far obstacles are from the self-driving car.

**Input:**

| Input Image                 | Depth Image                 | Segmented Image                 | Colored Segmented Image                 |
| --------------------------- | --------------------------- | ------------------------------- | --------------------------------------- |
| ![](images/input_image.png) | ![](images/depth_image.png) | ![](images/segmented_image.png) | ![](images/colored_segmented_image.png) |

## Step 1 : Drivable Space Estimation in 3D

Input: Output of a semantic segmentation neural networks.

### Step 1.1 - Estimating the x, y, and z coordinates of every pixel in the image

![](images/xyz.png)

### Step 1.2 - Estimating The Ground Plane Using [RANSAC](https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/FISHER/RANSAC/)

1. We need to process the semantic segmentation output to extract the relevant pixels belonging to the class you want consider as ground. For this project the **road** class as a **mapping index of 7.**

![](images/segmentation_mapping_index.png)

2. Use the extracted x, y, and z coordinates of pixels belonging to the road to estimate the ground plane.

| Road Mask                 | Ground Plane Mask           | Ground Plane Mask 3D (point cloud) |
| ------------------------- | --------------------------- | ---------------------------------- |
| ![](images/Road_Mask.png) | ![](images/Ground_Mask.png) | ![](images/Ground_Mask_3D.png)     |

We still needs to perform lane estimation to know where it is legally allowed to drive.

## Step 2 : Lane Estimation Using The Semantic Segmentation Output

### Step 2.1 - Estimate Lane Boundary Proposals

We can estimate any line that qualifies as a lane boundary (line proposals) using the output from semantic segmentation.

1. Create an image containing the semantic segmentation pixels belonging to categories relevant to the lane boundaries.
   - Pixels of Lane Markings: 6
   - Pixels of Side Walks: 8.
2. Perform edge detection on the derived lane boundary image. `cv2.Canny()`
3. Perform line estimation on the output of edge detection. `cv2.HoughLinesP()`

![](images/Lane_Markings.png)

### Step 2.2 - Merge and Filter Lane Lines

We need now to merge redundant lines, and filter out any horizontal lines apparent in the image.
- To merge redundant lines -> group lines with similar slope and intercept.
- To filter horizontal lines -> slope thresholding.

![](images/Lane_Markings_Merged_and_Filtered.png)

Which gives us one line per lane.

So now we can select the lane markings belonging to the current lane.

![](images/Current_Lane_Markings.png)


## Step 3 : Computing Minimum Distance To Impact Using The Output of 2D Object Detection.

We need to determine the minimum distance to impact with objects in the scene.

Objects identified on the scene:

![](images/Detections.png)

### Step 3.1 - Filtering Out Unreliable Detections

This incorrect detections are caused by a high precision, low recall object detector. To solve this problem, the output of the semantic segmentation network has to be used to eliminate unreliable detections.

![](images/Filtered_Detections.png)


### Step 3.2 - Estimating Minimum Distance To Impact

This can be performed by simply taking the minimum distance from the pixels in the bounding box to the camera center.
![](images/Estimated_Distance.png)


## :bulb: What we should remember
- The output of semantic segmentation can be used to estimate drivable space. 
- Classical computer vision can be used to find lane boundaries. 
- The output of semantic segmentation can be used to filter out unreliable output from object detection. 
