# LOAD THE DATASETS
import cv2 
import os
import glob
import torch
import random
import numpy
torch.set_default_tensor_type(torch.FloatTensor)

#img_dir = "./face_images/*.jpg"
img_dir = "./face_images/image00000.jpg"
files = glob.glob(img_dir)
data = []

for fl in files:
    img = cv2.imread(fl)
    data.append(img) #img.T

# Load data into tensor of size nImages x Channels x Height x Width
    # nImages = number of images in the folder
    # Channels = 3 (RBG colors)
    # Height, Width = 128
imgTens= torch.tensor(data)

# Randomly shuffle the data using torch.randperm
index = torch.randperm(imgTens.shape[0])
imgTens = imgTens[index].view(imgTens.size())

# AUGMENT YOUR DATA
# Augment by a small factor such as 10 to reduce overfitting by using OpenCV to transform your original images
augImg = torch.cat((imgTens, imgTens, imgTens, imgTens, imgTens, imgTens, imgTens, imgTens, imgTens, imgTens), 0)

#TEST
currImg = imgTens[0]

#Horizontal Flips
cv2.imshow('Image', cv2.flip(currImg.numpy(), 1))
cv2.waitKey(0)

#Random Crops
newX = numpy.random.randint(0, 64)
newY = numpy.random.randint(0, 64)

'''
for i in range(imgTens.shape[0], augImg.shape[0]):
    currImg = augImg[i]

    # horizontal flips
    if random.random() < 0.5:
        currImg = torch.from_numpy(cv2.flip(currImg.numpy(), 0))

    # random crops
    newX = numpy.random.randint(0, 128)
    newY = numpy.random.randint(0, 128)

    # scalings of input RBG values by single scaler randomly chosen between [0.6, 1.0]

    augImg[i] = currImg
'''

# CONVERT YOUR IMAGES TO L * a * b * COLOR SPACE
    # image = cv2.imread('example.jpg')
    # imageLAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# BUILD A SIMPLE REGRESSOR
    # Using convolutional layers, that predict the mean chrominance values for the entire input image
    # Input: grayscale image (only the L* channel)
    # Output: predicts mean chrominance (take the mean across all pixels to obtain mean a* and mean b*) values across all pixels of the image, ignoring pixel location

# ONCE YOU HAVE THIS WORKING, MAKE A COPY OF THIS CODE SO THAT YOU CAN SUBMIT IT LATER.