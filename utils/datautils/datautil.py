import os
import cv2
import logging
from tqdm import tqdm

__logger = logging.getLogger(__name__)


def read_file(fpath, extension):
    exts = {
        "png": cv2.imread,
        "jpg": cv2.imread,
        "tif": cv2.imread,
    }
    try:
        return exts[extension](fpath)
    except KeyError as e:
        __logger.error(e)
    finally:
        pass


class DataUtil:
    @staticmethod
    def read_dir(dirpath, exts: list):
        num_files = sum([len(files) for r, d, files in os.walk(dirpath)])
        for root, __, files in tqdm(
            os.walk(dirpath), desc="reading files", total=num_files
        ):
            for f in files:
                file_ext = os.path.splitext(f)[-1].strip(".")
                if file_ext in exts:
                    yield read_file(os.path.join(root, f), file_ext)
