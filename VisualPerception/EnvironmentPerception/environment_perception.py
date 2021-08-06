import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt
from m6bk import *
from utils import plot_image

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

plt.show()
