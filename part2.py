import cv2
import os
import glob

img_dir = "/face_images"
files = glob.glob(img_dir)
data = []

for fl in files:
    img = cv2.imread(fl)
    data.append(img)