#!/usr/local/anaconda/bin/python2.7
# -*- coding: utf-8 -*-

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse
from skimage.color import rgb2lab
import numpy as np


def convert_graph(segments):
    num_labels = segments.max() + 1
    node = [ [] for label in range(0, num_labels)]
    graph = [ [] for label in range(0, num_labels)]
    # print len(graph)
    size = segments.shape
    for i in range(size[0]):
        for j in range(size[1]):
            label = segments[i, j]
            # print label
            node[label].append((i, j))

#    for element in node:
#        print element

    for i in range(size[0]):
        for j in range(size[1] - 1):
            label1 = segments[i, j]
            label2 = segments[i, j + 1]
            if label1 != label2:
                if label2 not in graph[label1]:
                    graph[label1].append(label2)

    for i in range(size[0]-1):
        for j in range(size[1]):
            label1 = segments[i, j]
            label2 = segments[i + 1, j]
            if label1 != label2:
                if label2 not in graph[label1]:
                    graph[label1].append(label2)

    for i in range(size[0] - 1):
        for j in range(size[1] - 1):
            label1 = segments[i, j]
            label2 = segments[i + 1, j + 1]
            if label1 != label2:
                if label2 not in graph[label1]:
                    graph[label1].append(label2)

    for i in range(0, size[0] - 1):
        for j in range(1, size[1]):
            label1 = segments[i, j]
            label2 = segments[i + 1, j - 1]
            if label1 != label2:
                if label2 not in graph[label1]:
                    graph[label1].append(label2)

            # print label

#    for element in graph:
#        print element

    return graph, node


def distance(point1, point2):
    return ((point1 - point2)**2).sum()


def average_superpixel(node):
    superpixels = []
    average = []
    for element in node:
        row = []
        col = []
        for each_lo in element:
            row.append(each_lo[0])
            col.append(each_lo[1])
        superpixel = image[row, col, :]
        superpixel = superpixel.reshape(len(element), 1, 3)
        superpixels.append(superpixel)
        sp_lab = rgb2lab(superpixel)
        # print sp_lab
        # print sp_lab.mean(0)
        average.append(sp_lab.mean(0))
    return average, superpixels


def get_pairs(dis, thresh, graph):
    pairs = []
    for i in range(len(dis)):
        mini_distance = 10000
        for j in range(len(dis[i])):
            if dis[i][j] < mini_distance:
                mini_distance = dis[i][j]
                index = j
        if mini_distance < thresh:
            pairs.append([i, graph[i][index]])

    return pairs


def findSet(pairs):
    node1 = pairs[0][0]
    node2 = pairs[0][1]
    new_pairs = []
    nodeSet = [node1, node2]
    for element in pairs:
        flag = 0
        for i in nodeSet:
            if i in element:
                flag = 1
                for num in element:
                    if num not in nodeSet:
                        nodeSet.append(num)
                break
        if flag == 0:
            new_pairs.append(element)
    return nodeSet, new_pairs


def update(graph, node, pairs, size):
    new_node = []
    new_graph = []
    label = 0
    segment = np.zeros(size, dtype=np.int)
    while len(pairs) != 0:
        nodeSet, pairs = findSet(pairs)
        # print nodeSet
        for i in range(len(nodeSet)):
            locations = node[nodeSet[i]]
            for l in locations:
                # print l
                segment[l[0]][l[1]] = label
        label = label + 1

    # print segment
    # fig = plt.figure("new segment")
    # ax = fig.add_subplot(1, 1, 1)
    # ax.imshow(mark_boundaries(image, segment, color=(1, 0, 0), mode='outer'))

    return segment


def merge_supixel(graph, node, image):
    dis = []
    average, superpixels = average_superpixel(node)
    # print superpixels[58]
    # print average[58]
    # size =  superpixels[58].shape
    # fig = plt.figure("superpixel:")
    # ax = fig.add_subplot(1, 1, 1)
    # ax.imshow(superpixels[58])

    for label in range(0, len(graph)):
        dis.append([distance(average[label], average[neighbor]) for neighbor in graph[label]])

    print dis
    pairs = get_pairs(dis, 10, graph)
    print pairs
    segment = update(graph, node, pairs, image.shape[0:2])

    return segment


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image and convert it to a floating point data type
image = img_as_float(io.imread(args["image"]))
image = image[1:15, 1:623, :]
# loop over the number of segments

numSegments = [50, 100, 200]

'''
fig1 = plt.figure("Superpixels segments")
ax = fig1.add_subplot(len(numSegments) + 1, 1, 1)
plt.axis("off")
ax.imshow(image)
'''
plt.figure("Superpixels")                # the first figure
plt.subplot(len(numSegments) + 1, 1, 1)
plt.imshow(image)
plt.axis("off")


for i in range(0, len(numSegments)):
    # apply SLIC and extract (approximately) the supplied number
    # of segments
    print numSegments[i]
    segments = slic(image, n_segments=numSegments[i], sigma=0.2)
    # print segments
    plt.subplot(len(numSegments) + 1, 1, i+2)
    # show the output of SLIC
    plt.imshow(mark_boundaries(image, segments, color=(1, 0, 0), mode='outer'))
    plt.axis("off")

graph, node = convert_graph(segments)
# print graph
# imagelab = rgb2lab(image)
plt.figure("Superpixels merge")

for i in range(4):
    imagemerg = merge_supixel(graph, node, image)
    # print imagemerg
    plt.subplot(len(numSegments) + 1, 1, 4 - i)
    plt.imshow(mark_boundaries(image, imagemerg, color=(1, 0, 0), mode='outer'))
    plt.axis("off")
    graph, node = convert_graph(imagemerg)


    # show the plots
plt.show()


