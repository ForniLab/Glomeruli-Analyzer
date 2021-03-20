import numpy as np
import cv2


class BGRUtil:
    __color_ranges = {
        "r": {"low": np.array([147, 5, 2]), "high": np.array([7, 255, 255])},
        "g": {"low": np.array([27, 5, 2]), "high": np.array([67, 255, 255])},
        "b": {"low": np.array([87, 5, 2]), "high": np.array([127, 255, 255])},
    }

    __channel = {
        "b": 0,
        "g": 1,
        "r": 2,
    }

    def __init__(self):
        super().__init__()

    def channel_num(self, color: str):
        """
        This is a utility object aimed to facilitate
        using color related settings.
        
        Arguments:
            color {String} -- can only be one of "R", "r", "G", "g", "B","b"
        """
        return self.__channel[color.lower()]

    def color_range(self, color: str, return_dict=False):
        """Gives the highmost and lowmost colors of a channel.
        Especially created to help when attempting to use
        cv2.inRange.
        
        Arguments:
            color {str} -- the color of the channel. 
        
        Keyword Arguments:
            return_dict {bool} -- a flag whether to return the dict or only the values
        
        Returns:
            [list of ndarrays] -- if return_dict is false
            [Dict] -- if return_dict is true
        """
        return (
            self.__color_ranges[color.upper()].values()
            if return_dict is False
            else self.__color_ranges[color.upper()]
        )

    def create_color_mask(self, img, color: str):
        lower, upper = self.color_range(color)
        if color.upper() == "R":
            # Opencv cannot handle the cyclic nature of R ranges
            # so we have to do this manually
            mask1 = cv2.inRange(img, lowerb=np.array((0, 5, 5)), upperb=upper)
            mask2 = cv2.inRange(
                img, lowerb=lower, upperb=np.array((179, 255, 255))
            )
            return mask1 + mask2
        else:
            return cv2.inRange(img, lower, upper)

    def contour_outline_color(self, c: str):
        return {"R": (0, 0, 255), "G": (0, 255, 0), "B": (255, 0, 0)}[
            c.upper()
        ]

    def channel_name(self, channel_id):
        for k, v in self.__channel.items():
            if v == channel_id:
                return k