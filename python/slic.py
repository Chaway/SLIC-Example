#!/usr/local/anaconda/bin/python2.7
# -*- coding: utf-8 -*-

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image and convert it to a floating point data type
image = img_as_float(io.imread(args["image"]))

# loop over the number of segments

numSegments = [50, 100, 200]
fig = plt.figure("Superpixels segments")
ax = fig.add_subplot(len(numSegments) + 1, 1, 1)
plt.axis("off")
ax.imshow(image)
for i in range(0, len(numSegments)):
    # apply SLIC and extract (approximately) the supplied number
    # of segments
    print numSegments[i]
    segments = slic(image, n_segments=numSegments[i], sigma=0.5)

    ax = fig.add_subplot(len(numSegments) + 1, 1, i+2)
    # show the output of SLIC
    ax.imshow(mark_boundaries(image, segments, color=(1, 0, 0), mode='outer'))
    plt.axis("off")

# show the plots
plt.show()