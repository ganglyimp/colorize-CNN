# LOAD THE DATASETS

import cv2
import os 
import glob
import torch
import random
import numpy as np
torch.set_default_tensor_type(torch.FloatTensor)

device = torch.device("cuda:0")

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
imgTens = torch.tensor(data)

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
        C = 128 #in/out channels
        K = 3 #kernel size

        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=C, kernel_size=K, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=C, out_channels=C, kernel_size=K, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=C, out_channels=C, kernel_size=K, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=C, out_channels=C, kernel_size=K, stride=2, padding=1)
        self.conv5 = nn.Conv2d(in_channels=C, out_channels=C, kernel_size=K, stride=2, padding=1).cuda()
        self.conv6 = nn.Conv2d(in_channels=C, out_channels=C, kernel_size=K, stride=2, padding=1).cuda()

        # Scales weights by gain parameter
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.conv4.weight)
        nn.init.xavier_uniform_(self.conv5.weight)
        nn.init.xavier_uniform_(self.conv6.weight)

        # Batch Normalization 
        self.norm1 = nn.BatchNorm2d(C)
        self.norm2 = nn.BatchNorm2d(C)
        self.norm3 = nn.BatchNorm2d(C)
        self.norm4 = nn.BatchNorm2d(C)
        self.norm5 = nn.BatchNorm2d(C)
        self.norm6 = nn.BatchNorm2d(C)

        # Deconvolutional Layers
        self.deconv1 = nn.ConvTranspose2d(in_channels=C, out_channels=C, kernel_size=K, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=C, out_channels=C, kernel_size=K, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=C, out_channels=C, kernel_size=K, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(in_channels=C, out_channels=C, kernel_size=K, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(in_channels=C, out_channels=C, kernel_size=K, stride=2, padding=1)
        self.deconv6 = nn.ConvTranspose2d(in_channels=C, out_channels=2, kernel_size=K, stride=2, padding=1)

    def forward(self, t):
        # (1) hidden conv layer
        t = self.conv1(t)
        t = self.norm1(t)
        t = F.relu(t)

        # (2) hidden conv layer
        t = self.conv2(t)
        t = self.norm2(t)
        t = F.relu(t)

        # (3) hidden conv layer
        t = self.conv3(t)
        t = self.norm3(t)
        t = F.relu(t)

        # (4) hidden conv layer
        t = self.conv4(t)
        t = self.norm4(t)
        t = F.relu(t)

        # (5) hidden conv layer
        t = self.conv5(t)
        t = self.norm5(t)
        t = F.relu(t)

        # (6) hidden conv layer
        t = self.conv6(t)
        t = self.norm6(t)
        t = F.relu(t)

        # Deconv Layers
        t = self.deconv1(t, output_size=(10,128,4,4))
        t = self.deconv2(t, output_size=(10,128,8,8))
        t = self.deconv3(t, output_size=(10,128,16,16))
        t = self.deconv4(t, output_size=(10,128,32,32))
        t = self.deconv5(t, output_size=(10,128,64,64))
        t = self.deconv6(t, output_size=(10,2,128,128))

        return t

network = Network()
network = network.to(device)

# Divide dataset into two parts: 90% training, 10% testing
inputMat = normalGreyImg.unsqueeze(1)
tenPercent = int(inputMat.size(dim=0) * 0.1)

testInput = inputMat[0:tenPercent, :, :, :]
trainInput = inputMat[tenPercent:, :, :, :]

testInput = testInput.to(device)
trainInput = trainInput.to(device)

testLabels = chromValues[0:tenPercent, :, :, :]
trainLabels = chromValues[tenPercent:, :, :, :]

testLabels = testLabels.to(device)
trainLabels = trainLabels.to(device)

# Creating mini-batches
batchSize = 10
trainLoader = torch.utils.data.DataLoader(trainInput, batch_size=batchSize)
labelLoader = torch.utils.data.DataLoader(trainLabels, batch_size=batchSize)

testLoader = torch.utils.data.DataLoader(testInput, batch_size=batchSize)
testLabelLoader = torch.utils.data.DataLoader(testLabels, batch_size=batchSize)

optimizer = optim.Adam(network.parameters(), lr= .01)
lossFunc = nn.MSELoss()

print("Training the network...")
for trainBatch in trainLoader:
    labelBatch = next(iter(labelLoader))

    pred = network(trainBatch)

    #Finding loss & calculating gradients
    loss = lossFunc(pred, labelBatch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Testing the network...")
#lastBatch = []
#lastPreds = []

i = 0
meanLoss = 0
network.eval()
for testBatch in testLoader:
    optimizer.zero_grad()
    labelBatch = next(iter(testLabelLoader))

    pred = network(testBatch)

    #Finding loss & calculating gradients
    loss = lossFunc(pred, labelBatch)
    meanLoss += loss.item()

    #Merging predicted A*B* channels for colorized image output (last batch only)
    #if i == 74:
        #lastBatch = testBatch.numpy()
        #lastPreds = pred.detach().numpy()
    
    i += 1

# Average MSE
meanLoss = meanLoss / i
print("Mean loss: ", meanLoss)

# Combining input L* channel and predicted A* B* channels to produce colorized image
'''
coloredImg = np.zeros((10, 128, 128, 3))

lastBatch = np.interp(lastBatch, (lastBatch.min(), lastBatch.max()), (0, 100))
lastPreds = np.interp(lastPreds, (lastPreds.min(), lastPreds.max()), (-128, 128))

for i in range(10):
    coloredImg[i, :, :, 0] = lastBatch[i, 0, :, :]
    coloredImg[i, :, :, 1] = lastPreds[i, 0, :, :]
    coloredImg[i, :, :, 2] = lastPreds[i, 1, :, :]

for i in range(10):
    pictureBGR = cv2.cvtColor(coloredImg[i].astype(np.float32), cv2.COLOR_LAB2BGR)
    imgName = "Colored Image " + str(i)
    cv2.imshow(imgName, pictureBGR) 

cv2.waitKey(0)
cv2.destroyAllWindows()
'''
