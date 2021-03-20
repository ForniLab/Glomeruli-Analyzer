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


CONTOUR_AREA_CUTOFF = 5


def enhance_image_contrast(img, clipLimit=2.0, tileGridSize=(4, 4)):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # -----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)
    # -----Applying CLAHE histogram noramlization
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    modified = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return modified


def detect_edges(single_ch_img):
    """
    Arguments:
        single_ch_img {numpy.ndarray} -- should be a 2D matrix.
    
    Returns:
        numpy.ndarray -- the edgy image
    """    
    filtered = 255 - single_ch_img
    filtered = cv2.morphologyEx(
        filtered, cv2.MORPH_OPEN, iterations=5, kernel=(3, 3)
    )
    ret, thresh = cv2.threshold(
        filtered, 240, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV
    )
    ret, thresh = cv2.threshold(filtered, ret, 255, cv2.THRESH_BINARY_INV)
    canny = cv2.Canny(thresh, 255 // 4, 255)
    cnts, hier = cv2.findContours(
        canny, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    return cnts, hier


def process_contours(img):
    """
    The main process function, 
    applies subprocess functions and detects the
    contours of a pictures.
    
    Arguments:
        img {[numpy.ndarray]} -- input image
    
    Returns:
        [numpy.ndarray] -- output mask
    """    
    bgr_util = BGRUtil()
    channels = list("rbg")
    contours = {c: [] for c in channels}
    hierarchies = {c: [] for c in channels}

    for ch in channels:
        n = bgr_util.channel_num(ch)
        segmented = img[:, :, n]
        cntrs, hiers = detect_edges(segmented)
        contours[ch].extend(cntrs)
        hierarchies[ch].extend(hiers)

    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    mask = np.zeros(img.shape).astype(np.uint8)

    for ch in channels:
        _conts = contours[ch]
        _hiers = hierarchies[ch]
        _conts = [
            x for x in _conts if cv2.contourArea(x) > CONTOUR_AREA_CUTOFF
        ]
        mask += cv2.drawContours(
            np.zeros(mask.shape).astype(np.uint8),
            _conts,
            -1,
            bgr_util.contour_color(ch),
            1,
        )

    return mask + img


def scale_hsv(h, s, v):
    return np.array((h // 2, (s * 255) // 100, (v * 255) // 100))


def show_img(ax, img, title, **kwargs):
    ax.imshow(img, **kwargs)
    ax.set_title(title)


if __name__ == "__main__":
    out_path = "outputs/hist/"

    img = cv2.imread(
        "../../dataset/original_dropbox_manipulated/manipulated_data/"
        + "P21_AP2e_0051_AOB_Het_Kirrel2-680 Kirrel3-568 GFP-488 8-13-19/"
        + "AOB2/MAX_Image 8.jpg"
    )
    mask = process_contours(img.copy())
    cv2.imwrite("result.png", img)

