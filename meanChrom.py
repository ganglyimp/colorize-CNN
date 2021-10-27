# LOAD THE DATASETS
import cv2 
import os
import glob
import torch
import random
import numpy as np
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
    cropSize = np.random.randint(64, 128)

    newX = np.random.randint(0, cropSize)
    newY = np.random.randint(0, cropSize)

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

# Input: Grey scale image (L channel only)
normalGreyImg = torch.zeros(7500, 128, 128)
meanChromTest = torch.zeros(7500, 2)
for i in range(7500):
    LChan, AChan, BChan = cv2.split(augImg[i].numpy())

    meanChromTest[i, 0] = np.mean(AChan)
    meanChromTest[i, 1] = np.mean(BChan)

    LChan = (LChan - np.min(LChan)) / (np.max(LChan) - np.min(LChan))
    normalGreyImg[i, :, :] = torch.from_numpy(LChan)

# preprepare mean chrom to test on

import torch.nn as nn

# BUILD A SIMPLE REGRESSOR
    # Using convolutional layers, that predict the mean chrominance values for the entire input image
    # Input: grayscale image (only the L* channel)
    # Output: predicts mean chrominance (take the mean across all pixels to obtain mean a* and mean b*) values across all pixels of the image, ignoring pixel location

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120) #fully connected layer
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    
    def forward(self, t):
        t = self.layer
        return t




# ONCE YOU HAVE THIS WORKING, MAKE A COPY OF THIS CODE SO THAT YOU CAN SUBMIT IT LATER.