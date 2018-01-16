import random
from typing import Tuple

import cv2
import numpy
import pandas
import matplotlib.pyplot as plt
from tqdm import tqdm


def IOU(x, centroids):
    similarities = []
    k = len(centroids)
    for centroid in centroids:
        c_w, c_h = centroid
        w, h = x
        if c_w >= w and c_h >= h:
            similarity = w * h / (c_w * c_h)
        elif c_w >= w and c_h <= h:
            similarity = w * c_h / (w * h + (c_w - w) * c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w * h / (w * h + c_w * (c_h - h))
        else:  # means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w * c_h) / (w * h)
        similarities.append(similarity)  # will become (k,) shape
    return numpy.array(similarities)


def avg_IOU(X, centroids):
    n, d = X.shape
    sum = 0.
    for i in range(X.shape[0]):
        # note IOU() will return array which contains IoU for each centroid and X[i] // slightly ineffective, but I am too lazy
        sum += max(IOU(X[i], centroids))
    return sum / n


def kmeans(X: numpy.ndarray, centroids: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
    N = X.shape[0]
    k, dim = centroids.shape
    prev_assignments = numpy.ones(N) * (-1)
    iter = 0
    old_D = numpy.zeros((N, k))

    while True:
        D = []
        iter += 1
        for i in range(N):
            d = 1 - IOU(X[i], centroids)
            D.append(d)
        D = numpy.array(D)  # D.shape = (N,k)
        mean_IOU = numpy.mean(D)

        # print("iter {}: mean iou = {}; dists = {}".format(iter, mean_IOU, numpy.sum(numpy.abs(old_D - D))))

        # assign samples to centroids
        assignments = numpy.argmin(D, axis=1)

        if (assignments == prev_assignments).all():
            return mean_IOU, centroids

        # calculate new centroids
        centroid_sums = numpy.zeros((k, dim), numpy.float)
        for i in range(N):
            centroid_sums[assignments[i]] += X[i]
        for j in range(k):
            centroids[j] = centroid_sums[j] / (numpy.sum(assignments == j))

        prev_assignments = assignments.copy()
        old_D = D.copy()


def visualize_anchors(anchors: numpy.ndarray, visualization_width: int = 1000, visualization_height: int = 1000):
    colors = [(255, 0, 0), (255, 255, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 100, 0),
              (0, 255, 100), (255, 255, 255), (100, 255, 55)]

    blank_image = numpy.zeros((visualization_height, visualization_width, 3), numpy.uint8)

    stride_h = 10
    stride_w = 10

    for i in range(len(anchors)):
        (w, h) = anchors[i]
        w = int(w * visualization_width)
        h = int(h * visualization_height)
        # print(w, h)
        left_upper_corner = (10 + i * stride_w, 10 + i * stride_h)
        right_lower_corner = (left_upper_corner[0] + w, left_upper_corner[1] + h)
        cv2.rectangle(blank_image, left_upper_corner, right_lower_corner, colors[i])

    cv2.imwrite("anchors-{0}.png".format(len(anchors)), blank_image)

    # cv2.namedWindow('Image')
    # cv2.imshow('Image', blank_image)
    # cv2.waitKey()


if __name__ == "__main__":
    annotation_dimensions = pandas.read_csv("data/bounding_box_dimensions_relative.csv")
    visualization_width, visualization_height = 3500, 1500

    # annotation_dimensions = pandas.read_csv("data/bounding_box_dimensions_cropped_images_relative.csv")
    # visualization_width, visualization_height = 600, 300

    total_number_of_clusters = 10

    annotation_dimensions.plot.scatter(x='width', y='height', s=0.1, c='red')
    plt.show()

    dims = annotation_dimensions[['width', 'height']].as_matrix()

    statistics = []

    for num_clusters in tqdm(range(1, total_number_of_clusters + 1)):
        indices = [random.randrange(dims.shape[0]) for i in range(num_clusters)]
        initial_centroids = dims[indices]
        meanIntersectionOverUnion, centroids = kmeans(dims, initial_centroids)
        statistics.append((num_clusters, meanIntersectionOverUnion, centroids))

    for (clusters, iou, centroids) in statistics:
        print("{0} clusters: {1:.4f} mean IOU".format(clusters, iou))
        for c in centroids:
            print("Ratio: {0:.4f} = {1:.0f}x{2:.0f} px scaled to {3}x{4} image".format(c[0] / c[1],
                                                                                       c[0] * visualization_width,
                                                                                       c[1] * visualization_height,
                                                                                       visualization_width,
                                                                                       visualization_height))

    for (clusters, iou, centroids) in statistics:
        visualize_anchors(centroids, visualization_width, visualization_height)
        print("{0} clusters: Centroids = {1}".format(clusters, centroids))
