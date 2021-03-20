import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from sklearn.cluster import KMeans

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append("../..")
    pass

from utils.color_utils.bgrutil import BGRUtil


def downscale(img, factor: int):
    return cv2.resize(
        img,
        (int(img.shape[1]/factor), int(img.shape[0]/factor)),
        interpolation=cv2.INTER_AREA,
    )


def upscale(img, factor: int):
    return cv2.resize(
        img,
        (int(img.shape[1] * factor), int(img.shape[0] * factor)),
        interpolation=cv2.INTER_CUBIC,
    )


def detect_contours(thresh: np.array) -> list:
    canny = cv2.Canny(thresh, 255 // 3, 255)
    canny = downscale(canny, 1.25)
    canny = upscale(canny, 1.25)
    contours, hier = cv2.findContours(
        canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    return contours, hier


def create_contour_mask(
    shape: tuple, contours: list, hierarchy=None, thickness=1
) -> np.array:
    hulls = [
        # cv2.approxPolyDP(
            # c, 0.001 * cv2.arcLength(c, closed=False), closed=True
        # )
        c for c in contours
    ]
    mask = np.zeros(shape, dtype=np.uint8)
    mask = cv2.drawContours(mask, hulls, -1, 255, thickness=thickness)
    return mask


def fill_contours(shape: tuple, contours: list):
    mask = create_contour_mask(shape, contours)
    for c in contours:
        cv2.fillPoly(mask, c, 255)
    return mask

def fill_contour(shape:tuple, c):
    mask = np.zeros(shape, dtype=np.uint8)
    mask = cv2.drawContours(mask, [c], -1, 255, cv2.FILLED)
    return mask

import random 

def process(img, ch, thickness=1):
    thresh = preprocess(img[:, :, ch].copy())
    contours, hier = detect_contours(thresh)
    mask = create_contour_mask(
        thresh.shape, contours, hier, thickness=thickness)
    return mask


def preprocess(segmented, morph_kernel=(7, 7), morph_iter=5):
    segmented = cv2.fastNlMeansDenoising(segmented, h=40)
    segmented = cv2.GaussianBlur(segmented, (7, 7), 21)
    segmented = cv2.morphologyEx(
        segmented, cv2.MORPH_CLOSE, morph_kernel, iterations=morph_iter
    )
    thresh = cv2.adaptiveThreshold(
        segmented,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        299,
        0.2,
    )
    thresh = 255 - thresh
    plt.imshow(thresh, "gray")
    plt.show()

    return thresh


if __name__ == "__main__":
    out_path = "outputs/hist/"
    img = cv2.imread("1.png")    
    mask = process(img, 2, thickness=cv2.FILLED)
    Image.fromarray(mask).show()
