import os
import random
import shutil
import sys
from collections import Counter, defaultdict
import tqdm


import cv2
import matplotlib.pyplot as plt
import numpy as np
from hdbscan import HDBSCAN
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from scipy.cluster.vq import kmeans2
from sklearn.cluster import DBSCAN, KMeans


if __name__ == "__main__":
    # To switch to the directory of the script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append("../../")
    print(os.path.abspath(os.curdir))

from utils.datautils.datautil import DataUtil


def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(
        int(color[0]), int(color[1]), int(color[2])
    )


def get_top_colors(cutoff, X):
    # -- find the top k colors per cluster --
    top_colors = defaultdict(list)

    for c_lab, prob, color in zip(
        clusterer.labels_,
        clusterer.probabilities_,
        set([tuple(row) for row in X]),
    ):
        top_colors[c_lab].append((color, prob))

    for c_lab in top_colors:
        top_colors[c_lab].sort(reverse=True, key=lambda x: x[1])
        for idx, item in enumerate(top_colors[c_lab]):
            if item[1] < cutoff:
                top_colors[c_lab] = top_colors[c_lab][:idx]
                break

    return top_colors


if __name__ == "__main__":
    # To switch to the directory of the script
    out_path = "color_distribution"
    data_path = "../../dataset"

    # Creates the output dir afresh
    if os.path.isdir(out_path):
        shutil.rmtree(out_path)
    os.mkdir(out_path)

    # This part creates the dataset
    dataset = []
    for im in DataUtil.read_dir(data_path, ["jpg"]):
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im.reshape(-1, 3)
        # We don't need all of the pixels to find the
        # dominant colors, hence a simple sampling will be
        # sufficient.
        indices = np.random.choice(im.shape[0], 5 * 10 ** 4, replace=False)
        im = im[indices, :]
        dataset.extend(im)

    X = np.array(dataset)
    print("Dataset shape", X.shape)
    del dataset
    clusterer = KMeans(n_clusters=10, n_jobs=-1)
    clusterer.fit(X)

    # To create the pie chart by using centroids colors as cluster colors
    cntr = Counter(clusterer.labels_)
    vals = [cntr[i] for i in set(clusterer.labels_)]
    rgb_colors = [x for x in clusterer.cluster_centers_]
    hex_colors = [RGB2HEX(i) for i in rgb_colors]
    plt.figure(figsize=(20, 20))
    plt.pie(vals, labels=hex_colors, colors=hex_colors)
    plt.savefig(os.path.join(out_path, "color_clusters.png"))
    plt.close()
