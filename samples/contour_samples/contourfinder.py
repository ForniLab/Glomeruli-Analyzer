import os

import cv2
import numpy as np


if __name__ == "__main__":

    # Startup settings
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    os.chdir(script_dir)
    image_path = "source.tif"

    alpha = 5
    beta = 0
    threshold = 10

    mean = 0
    var = 3
    sigma = var ** 2

    # The main logic
    img = cv2.imread(image_path)
    out = img + np.random.normal(mean, sigma, img.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)

    out = cv2.fastNlMeansDenoisingColored(
        src=out, h=21, hColor=20, templateWindowSize=7, searchWindowSize=15
    )
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    out = cv2.bilateralFilter(out, 15, 75, 75)
    out = cv2.medianBlur(out, 15)

    out = cv2.dilate(out, (21, 21), iterations=2)
    out = cv2.erode(out, (21, 21), iterations=2)

    out = np.uint8(out)
    out = cv2.Canny(out, threshold, threshold * 2)

    contours, hierarchy = cv2.findContours(
        out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    hulls = [None] * len(contours)

    minEllipse = []
    for i, c in enumerate(contours):
        h = cv2.convexHull(contours[i])
        hulls[i] = h
        if h.shape[0] > 5:
            minEllipse.append(cv2.fitEllipse(h))

    im = img.copy()

    im = cv2.drawContours(img, hulls, contourIdx=-1, color=255, thickness=1)

    cv2.imwrite("result.png", im)

