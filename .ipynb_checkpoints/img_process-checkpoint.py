import os
import shutil
import cv2
import random
import csv

# set test image
im = cv2.imread("training/img/img_4.jpg")

imgheight=im.shape[0]
imgwidth=im.shape[1]

# Set image size of data points
M = 32
N = 32

shutil.rmtree("training/train")
os.mkdir("training/train")

row_list = [["id", "avg_R", "avg_G", "avg_B"]]

# Create random samples of M*N sized images
for img_id in range(100):
    y = random.randint(0, imgheight - M)
    x = random.randint(0, imgwidth - N)

    tiles = im[y:y+M,x:x+N]
    
    # average out rgb in image
    avg_rgb = tiles.mean(axis=0).mean(axis=0)
    row_list.append([img_id, avg_rgb[0], avg_rgb[1], avg_rgb[2]])

    # save image
    cv2.imwrite(f"training/train/{img_id}.png",tiles)

# Write data to csv
with open('training/train/train_data.csv', 'w', newline='\n') as file:
    writer = csv.writer(file)
    writer.writerows(row_list)