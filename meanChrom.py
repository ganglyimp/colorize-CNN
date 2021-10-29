# LOAD THE DATASETS
import cv2 
import os
import glob
import torch
import random
import numpy as np
torch.set_default_tensor_type(torch.FloatTensor)

print("Loading dataset...")

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
print("Preprocessing dataset...")

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

# BUILD A SIMPLE REGRESSOR
    # Using convolutional layers, that predict the mean chrominance values for the entire input image
    # Input: grayscale image (only the L* channel)
    # Output: predicts mean chrominance (take the mean across all pixels to obtain mean a* and mean b*) values across all pixels of the image, ignoring pixel location

# Network with 7 Modules, each module consists of a SpatialConvolution layer followed by a ReLU activation function
# SpatialConvolution layer: set padding & stride so that image after convolution is exactly half the size of the input
    # Decreasing powers of two: 128, 64, 32, 16
# Use a small number of feature maps (3) in the hidden layers

#kernel_size: sets filter size
#in_channels: # of color channels in input image
#out_channels: sets number of filers. one filter produces one output channel (feature maps)
#out_features: sets size of output tensor

import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as tf
import torch.optim as optim

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        C = 128
        K = 3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=C, kernel_size=K, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=C, out_channels=C, kernel_size=K, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=C, out_channels=C, kernel_size=K, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=C, out_channels=C, kernel_size=K, stride=2, padding=1)
        self.conv5 = nn.Conv2d(in_channels=C, out_channels=C, kernel_size=K, stride=2, padding=1)
        self.conv6 = nn.Conv2d(in_channels=C, out_channels=C, kernel_size=K, stride=2, padding=1)
        self.conv7 = nn.Conv2d(in_channels=C, out_channels=2, kernel_size=K, stride=2, padding=1)
    
    def forward(self, t):
        # (1) hidden conv layer
        t = self.conv1(t)
        t = F.relu(t)

        # (2) hidden conv layer
        t = self.conv2(t)
        t = F.relu(t)

        # (3) hidden conv layer
        t = self.conv3(t)
        t = F.relu(t)

        # (4) hidden conv layer
        t = self.conv4(t)
        t = F.relu(t)

        # (5) hidden conv layer
        t = self.conv5(t)
        t = F.relu(t)

        # (6) hidden conv layer
        t = self.conv6(t)
        t = F.relu(t)

        # (7) hidden conv layer
        t = self.conv7(t)
        t = F.relu(t)

        return t

network = Network()

#DEBUG FUNCTION. REMOVE BEFORE SUBMISSION
def correct(preds, labels): 
    return preds.argmax(dim=1).eq(labels).sum().item()

batchSize = 100

inputMat = normalGreyImg.unsqueeze(1)
trainLoader = torch.utils.data.DataLoader(inputMat, batch_size=batchSize)
labelLoader = torch.utils.data.DataLoader(meanChromTest, batch_size=batchSize)

print("Generating predictions...")

optimizer = optim.SGD(network.parameters(), lr= .1, momentum = .9)
lossFunc = nn.MSELoss()

totCorr = 0
for i in range(1, 7):
    optimizer.zero_grad()
    trainBatch = next(iter(trainLoader))
    labelBatch = next(iter(labelLoader))

    pred = network(trainBatch) #outputs a Nx2x1x1 tensor of tensors
    corr = correct(pred, labelBatch)
    totCorr += corr

    #Finding loss & calculating gradients
    pred = torch.reshape(pred, (batchSize, 2)) #reshape to same shape at labelBatch
    loss = lossFunc(pred, labelBatch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(i, " Loss: ", loss.item())

print("Correct: ", totCorr)



# https://deeplizard.com/learn/video/0VCOG8IeVf8

# ONCE YOU HAVE THIS WORKING, MAKE A COPY OF THIS CODE SO THAT YOU CAN SUBMIT IT LATER.