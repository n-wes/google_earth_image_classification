import pandas as pd
import numpy as np
import cv2
import sys
from sklearn.neighbors import KNeighborsClassifier
from joblib import load

def main(im_name, output_name):
    # load classifier
    clf = load('knn_classifier.joblib')

    # load image
    im = cv2.imread(im_name)
    overlay = im.copy()

    imgheight = im.shape[0]
    imgwidth = im.shape[1]

    # set image size of data points
    # for accuracy, this should match with size of test samples
    M = 32
    N = 32

    for y in range(0, imgheight, M):
        for x in range(0, imgwidth, N):
            y1 = y+M
            x1 = x+N

            tiles = im[y:y1,x:x1]
            avg_rgb = [tiles.mean(axis=0).mean(axis=0)]
            
            # check for deforested area
            if clf.predict(avg_rgb) == ['Deforested']:
                cv2.rectangle(im, (x,y), (x1,y1), (0,0,255), -1)

    # add transparency overlay for rectangles
    alpha = 0.8
    im_new = cv2.addWeighted(overlay, alpha, im, 1 - alpha, 0)

    # save image
    cv2.imwrite(f"results/{output_name}", im_new)

if __name__ == '__main__':
    main(sys.argv[1],sys.argv[2])