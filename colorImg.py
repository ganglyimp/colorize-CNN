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
    data.append(img)

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
chromValues = torch.zeros(7500, 2, 128, 128)
for i in range(7500):
    LChan, AChan, BChan = cv2.split(augImg[i].numpy())

    chromValues[i, 0] = torch.from_numpy(AChan)
    chromValues[i, 1] = torch.from_numpy(BChan)

    LChan = (LChan - np.min(LChan)) / (np.max(LChan) - np.min(LChan))
    normalGreyImg[i, :, :] = torch.from_numpy(LChan)

# BUILD A SIMPLE REGRESSOR
    # Using convolutional layers, that predict the mean chrominance values for the entire input image
    # Input: grayscale image (only the L* channel)
    # Output: predicts mean chrominance (take the mean across all pixels to obtain mean a* and mean b*) values across all pixels of the image, ignoring pixel location

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
        #self.conv6 = nn.Conv2d(in_channels=C, out_channels=C, kernel_size=K, stride=2, padding=1)
        #self.conv7 = nn.Conv2d(in_channels=C, out_channels=C, kernel_size=K, stride=2, padding=1)

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.conv4.weight)
        nn.init.xavier_uniform_(self.conv5.weight)
        #nn.init.xavier_uniform_(self.conv6.weight)
        #nn.init.xavier_uniform_(self.conv7.weight)

        #image size output = image size input, 2 color channels (2x128x128)
        #reduce downsampling layers to N, then also use N upsampling layers
        #start with N = 5 and experiment from there
        #initial spatial resolutions: 128, 64, 32, 16, 8, 4, 8, 16, 32, 64, 128
        self.deconv1 = nn.ConvTranspose2d(in_channels=C, out_channels=C, kernel_size=K, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=C, out_channels=C, kernel_size=K, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=C, out_channels=C, kernel_size=K, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(in_channels=C, out_channels=C, kernel_size=K, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(in_channels=C, out_channels=C, kernel_size=K, stride=2, padding=1)
        #self.deconv6 = nn.ConvTranspose2d(in_channels=C, out_channels=2, kernel_size=K, stride=2, padding=1)
        #self.deconv7 = nn.ConvTranspose2d(in_channels=C, out_channels=2, kernel_size=K, stride=2, padding=1)

        #after each Conv2d layer, insert SpatialBatchNormalization layer
        #requires 4D tensor inputs, so have to divide training dataset into mini-batches of say 10 images each
        #input: (NBatchx1xHEIGHTxWIDTH), output: (NBatchx2xHEIGHTxWIDTH)
        '''
        self.norm1 = nn.BatchNorm2d(C)
        self.norm2 = nn.BatchNorm2d(C)
        self.norm3 = nn.BatchNorm2d(C)
        self.norm4 = nn.BatchNorm2d(C)
        self.norm5 = nn.BatchNorm2d(C)
        self.norm6 = nn.BatchNorm2d(C)
        self.norm7 = nn.BatchNorm2d(C)
        '''


    def forward(self, t):
        # (1) hidden conv layer
        t = self.conv1(t)
        #t = self.norm1(t)
        t = F.relu(t)

        # (2) hidden conv layer
        t = self.conv2(t)
        #t = self.norm2(t)
        t = F.relu(t)

        # (3) hidden conv layer
        t = self.conv3(t)
        #t = self.norm3(t)
        t = F.relu(t)

        # (4) hidden conv layer
        t = self.conv4(t)
        #t = self.norm4(t)
        t = F.relu(t)

        # (5) hidden conv layer
        t = self.conv5(t)
        #t = self.norm5(t)
        t = F.relu(t)

        t = self.deconv1(t)
        t = self.deconv2(t)
        t = self.deconv3(t)
        t = self.deconv4(t)
        t = self.deconv5(t)

        return t

network = Network()

# Divide dataset into two parts: 90% training, 10% testing
inputMat = normalGreyImg.unsqueeze(1)
tenPercent = int(inputMat.size(dim=0) * 0.1)

testInput = inputMat[0:tenPercent, :, :, :]
trainInput = inputMat[tenPercent:, :, :, :]

testLabels = chromValues[0:tenPercent, :, :, :]
trainLabels = chromValues[tenPercent:, :, :, :]

# Creating mini-batches
batchSize = 10
trainLoader = torch.utils.data.DataLoader(trainInput, batch_size=batchSize)
labelLoader = torch.utils.data.DataLoader(trainLabels, batch_size=batchSize)

print("Generating predictions...")

optimizer = optim.Adam(network.parameters(), lr= .01)
lossFunc = nn.MSELoss()

#TEMP DEBUG LINE
pred = network(next(iter(trainLoader)))
print(pred.shape)

'''
i = 0
for trainBatch in trainLoader:
    optimizer.zero_grad()
    labelBatch = next(iter(labelLoader))

    pred = network(trainBatch)

    #Finding loss & calculating gradients
    loss = lossFunc(pred, labelBatch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    i+=1
'''

# To evaluate test images, print numerical mean square error value
# Also, run input luminance of image through network, 
# then merge a* and b* values predicted by regressor with input luminance, 
# and convert back to RGB color space (will let you view colorized images)