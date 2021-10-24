# LOAD THE DATASETS
import cv2 
import os
import glob
import torch
import random
import numpy
torch.set_default_tensor_type(torch.FloatTensor)

img_dir = "./face_images/*.jpg"
files = glob.glob(img_dir)
data = []

for fl in files:
    img = cv2.imread(fl)
    data.append(img) #img.T to get in project spec format

# Load data into tensor of size nImages x Channels x Height x Width
    # nImages = number of images in the folder
    # Channels = 3 (RBG colors)
    # Height, Width = 128

# Here's an issue. OpenCV expects images to be in format nImages x Height x Width x Channels
    # Does it affect anything having the dimensions being out of order? 
    # We at least need to use the OpenCV ordering for preprocessing.
imgTens= torch.tensor(data)

# Randomly shuffle the data using torch.randperm
index = torch.randperm(imgTens.shape[0])
imgTens = imgTens[index].view(imgTens.size())

# AUGMENT YOUR DATA
# Augment by a small factor such as 10 to reduce overfitting by using OpenCV to transform your original images
# there must be a better way of doing this than what I have going on. This is just ugly.
augImg = torch.cat((imgTens, imgTens, imgTens, imgTens, imgTens, imgTens, imgTens, imgTens, imgTens, imgTens), 0)

for i in range(imgTens.shape[0], augImg.shape[0]):
    currImg = augImg[i].numpy()

    # horizontal flips - 50% chance
    if random.random() < 0.5:
        currImg = cv2.flip(currImg, 0)

    # random crops - crop size ranges from 64 to 128
    cropSize = numpy.random.randint(64, 128)

    newX = numpy.random.randint(0, cropSize)
    newY = numpy.random.randint(0, cropSize)

    cropped = currImg[newY: newY + cropSize, newX: newX + cropSize]
    currImg = cv2.resize(cropped, (128, 128))

    # scalings of input RBG values by single scaler randomly chosen between [0.6, 1.0]
    randScalar = random.uniform(0.6, 1)
    for i in range(3):
        currImg[:, :, i] = currImg[:, :, i] * randScalar

    augImg[i] = torch.from_numpy(currImg)

# CONVERT YOUR IMAGES TO L * a * b * COLOR SPACE
for i in range(augImg.shape[0]):
    augImg[i] = torch.from_numpy(cv2.cvtColor(augImg[i].numpy(), cv2.COLOR_BGR2LAB))


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