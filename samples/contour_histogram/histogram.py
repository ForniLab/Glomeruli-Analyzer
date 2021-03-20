import os
import sys
import time
import warnings

import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np

# ----------------------------------
if __name__ == "__main__":

    # Startup settings
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    os.chdir(script_dir)
    sys_path = os.path.abspath("../..")
    sys.path.append(sys_path)
# ----------------------------------

from utils.color_utils.bgrutil import BGRUtil
from utils.contour_utils.contourutil import ContourUtil

warnings.filterwarnings(
    "ignore", message="invalid value encountered in greater_equal"
)
warnings.filterwarnings(
    "ignore", message="invalid value encountered in less_equal"
)


times = {}
MIN_CONTOUR_AREA = 10
if __name__ == "__main__":
    times["program_start"] = time.time()
    img_path = (
        "../../dataset/original_dropbox_manipulated/manipulated_data/"
        + "P21_AP2e_0051_AOB_Het_Kirrel2-680 Kirrel3-568 GFP-488 8-13-19/"
        + "AOB2/MAX_Image 8.jpg"
    )
    img = cv2.imread(img_path)
    print("input shape is: {}".format(img.shape))
    bgr = BGRUtil()
    contour_util = ContourUtil()
    channels = list("RGB")

    contours = []
    for ch_name in channels:
        c = bgr.channel_num(ch_name)
        single_chan = img[:, :, c]
        # ret, thresh = cv2.threshold(single_chan, 30, 255, cv2.THRESH_BINARY)
        thresh = cv2.adaptiveThreshold(
            single_chan,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=199,
            C=-5,
        )
        chan_contours, hier = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours += [
            c for c in chan_contours if cv2.contourArea(c) > MIN_CONTOUR_AREA
        ]

    times["contour_completion"] = time.time()

    for idx, c in enumerate(contours):
        if idx == 0:
            masked = contour_util.create_contour_mask(img, c)
            times["mask generation"] = time.time()
            c_sum = contour_util.compute_sum(masked)
            times["sum in channel"] = time.time()
            c_avg = contour_util.compute_average(masked)
            times["average in channel"] = time.time()
            c_hists = contour_util.compute_histogram(img, num_bins=50)
            times["generating hists"] = time.time()
            fig, axes = plt.subplots(3)
            for i in range(img.shape[2]):
                hist, bins = c_hists[i]
                width = 0.3 * (bins[1] - bins[0])
                center = (bins[:-1] + bins[1:]) / 2
                axes[i].bar(
                    center,
                    hist,
                    align="center",
                    width=width,
                    color=bgr.channel_name(i),
                )
                axes[i].set_title(
                    "Intensity Histogram in Channel {}".format(
                        bgr.channel_name(i)
                    )
                )
                axes[i].set_xticks(bins[::3])
            plt.subplots_adjust(hspace=1)
            plt.show()
    times = [(k, v) for k, v in times.items()]
    messeges = [
        "took {}us to compute {}.".format(
            times[idx][1] - times[idx - 1][1], times[idx][0]
        )
        for idx in range(len(times))
        if idx > 0
    ]

    for msg in messeges:
        print(msg)
