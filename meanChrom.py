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

# BUILD A SIMPLE REGRESSOR
    # Using convolutional layers, that predict the mean chrominance values for the entire input image
    # Input: grayscale image (only the L* channel)
    # Output: predicts mean chrominance (take the mean across all pixels to obtain mean a* and mean b*) values across all pixels of the image, ignoring pixel location

# ONCE YOU HAVE THIS WORKING, MAKE A COPY OF THIS CODE SO THAT YOU CAN SUBMIT IT LATER.