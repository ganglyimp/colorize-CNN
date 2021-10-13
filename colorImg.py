# LOAD THE DATASETS
import cv2 #OpenCV
import os
import glob

img_dir = "/face_images"
files = glob.glob(img_dir)
data = []

for fl in files:
    img = cv2.imread(fl)
    data.append(img)

# Should create a tensor of size nImages x Channels x Height x Width
    # nImages = number of images in the folder
    # Channels = 3 (RBG colors)
    # Height, Width = 128

# Load your data in a Tensor and randomly shuffle the data using torch.randperm
    # To reduce memory requirements, set default Torch datatype to 32-bit float with the 
    # following command at top of your program (before calling loader): torch.setdefaulttensortype('torch.FloatTensor')


# AUGMENT YOUR DATA
# Augment by a small factor such as 10 to reduce overfitting by using OpenCV to transform your original images

# CONVERT YOUR IMAGES TO L * a * b * COLOR SPACE
    # image = cv2.imread('example.jpg')
    # imageLAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# COLORIZE THE IMAGE

# BATCH NORMALIZATION
    # Incorporate batch normalization
    # This can be inserted directly after each SpatialConvolution layer

# TRAINING
    # Divide dataset into two parts: 
        # 90%: training
        # 10%: testing, make sure testing images have not been subjected to data augmentation
    # Train regressor on training part, and then test on the testing part
    # To evaluate the test images, print a numerical mean square error value.

# GPU COMPUTING
    # Speed up your network by moving your CNN to the GPU.
    # Will need to add commands for CUDA and cuNN