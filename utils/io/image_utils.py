import tifffile as tiff
import numpy as np
import pathlib
import cv2


class ImageUtils:
    def open(self, path):
        ext = pathlib.Path(path).suffix
        if ext.lower() in [".png", ".jpg"]:
            return cv2.imread(path)
        elif ext.lower() in [".tiff", ".tif"]:
            # Blue is the 2nd, Green is the 1st, Red is the last
            return tiff.imread(path).transpose((1,2,0))[:,:,[1,0,3]]
        else:
            raise Exception("Inavlid file format.")
