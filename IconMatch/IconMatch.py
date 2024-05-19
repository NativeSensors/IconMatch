import cv2 as cv

from PIL import ImageGrab
from IconMatch.box import grayscale_blur, canny_detection, group_rects

class ScreenScanner:
    """
    ScreenScanner class captures a screenshot, processes it, and detects rectangles
    using an image scanning algorithm.

    Attributes:
        scanner (ImageScanner): An instance of ImageScanner for processing images.
        thresh (int): The threshold value for image processing.
    """

    def __init__(self, thresh: int = 100):
        """
        Initializes the ScreenScanner with a specified threshold.

        Args:
            thresh (int): The threshold value for image processing. Default is 100.
        """
        self.scanner = ImageScanner(thresh)
        self.thresh = thresh

    def updateThresh(self, thresh: int):
        """
        Updates the threshold value used by the scanner.

        Args:
            thresh (int): The new threshold value.
        """
        self.scanner.updateThresh(thresh)

    def scan(self, bbox=None):
        """
        Captures a screenshot, processes it to detect rectangles, and adjusts
        the coordinates based on the bounding box.

        Args:
            bbox (tuple, optional): The bounding box for the screenshot. Default is None.

        Returns:
            list: A list of rectangles detected in the format (x, y, width, height).
        """
        screenshot = ImageGrab.grab(bbox=bbox)
        screenshot.save("__tmp.png")
        src = cv.imread("__tmp.png")
        # TODO: add x and y offset to the result rectangles
        rects = self.scanner.scan(src)

        if bbox:
            x, y = bbox[0], bbox[1]
            rects = [(rect[0] + x, rect[1] + y, rect[2], rect[3]) for rect in rects]
        return rects


class ImageScanner:
    """
    ImageScanner class processes images to detect rectangles based on edge detection.

    Attributes:
        thresh (int): The threshold value for image processing.
    """

    def __init__(self, thresh: int = 100):
        """
        Initializes the ImageScanner with a specified threshold.

        Args:
            thresh (int): The threshold value for image processing. Default is 100.
        """
        self.thresh = thresh

    def updateThresh(self, thresh: int):
        """
        Updates the threshold value used for image processing.

        Args:
            thresh (int): The new threshold value.
        """
        self.thresh = thresh

    def scan(self, src) -> list:
        """
        Processes an input image to detect rectangles.

        Args:
            src (MatLike): The source image to be processed.

        Returns:
            list: A list of bounding rectangles detected in the format (x, y, width, height).
        """
        # accept an input image and convert it to grayscale, and blur it
        gray_scale_image = grayscale_blur(src)

        # determine the bounding rectangles from canny detection
        _, bound_rect = canny_detection(gray_scale_image, min_threshold=self.thresh)

        # group the rectangles from this step
        grouped_rects = group_rects(bound_rect, 0, src.shape[1])

        return grouped_rects
