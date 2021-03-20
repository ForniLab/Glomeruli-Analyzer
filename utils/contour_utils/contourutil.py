import cv2
import numpy as np

# A Class which contains codes for
# contour related computations
class ContourUtil:
    def __init__(self):
        super().__init__()

    def compute_sum(self, masked: np.array) -> np.array:
        res = masked.reshape(-1, masked.shape[2])
        return np.sum(res, axis=0)

    def compute_average(
        self, masked: np.array, only_nonzero=False
    ) -> np.array:
        res = masked.reshape(-1, masked.shape[2])
        if only_nonzero == True:
            return np.true_divide(np.sum(res, axis=0), (res != 0).T.sum(1))
        else:
            return np.average(res, axis=0)

    def compute_histogram(
        self, img: np.array, num_bins=10, is_dense=True, nonzero=True
    ) -> list:
        """computes histogram for each channel in the image
        and returns the histograms in a list.

        Arguments:
            img {np.array} -- the multichannel image

        Keyword Arguments:
            num_bins {int} -- number of bins in the histogram (default: {10})
            is_dense {bool} -- whether the histogram will be a density plot or not
                                recommended to be True. (default: {True})

        Returns:
            list -- a list of histograms
        """
        hist_bins = []
        copy = img.copy().astype(np.float)
        if nonzero == True:
            copy[copy == 0] = float("nan")
        for i in range(copy.shape[2]):
            hist_bins.append(
                np.histogram(
                    copy[:, :, i],
                    bins=num_bins,
                    range=(0, 255),
                    density=is_dense,
                )
            )
        return hist_bins

    def create_contour_mask(self, img: np.array, contour):
        """[summary]
        Arguments:
            contour {list} -- a set of points which describe a contour
            shape {tuple} -- the shape of the output mask
        """
        mask = np.zeros(img.shape, dtype=np.uint8)
        mask = cv2.drawContours(mask, [contour], -1, (1, 1, 1), cv2.FILLED)
        mask = img * mask
        return mask

    def cropped_contour_mask(self, img: np.array, contour):
        x, y, w, h = cv2.boundingRect(contour)
        mask = np.zeros_like(img)
        mask = cv2.drawContours(mask, [contour], -1, (1, 1, 1), cv2.FILLED)
        return mask[y : y + h, x : x + w] * img[y : y + h, x : x + w]
