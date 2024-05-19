import cv2 as cv

from PIL import ImageGrab
from cv2.typing import MatLike
from icondetection.box import grayscale_blur, canny_detection, group_rects, candidate_rectangle
from icondetection.rectangle import Rectangle

class ScreenScanner:

    def __init__(self,thresh : int = 100):

        self.scanner = ImageScanner(thresh)
        self.thresh = thresh


    def scan(self,bbox = None):

        screenshot = ImageGrab.grab(bbox = bbox)
        screenshot.save("__tmp.png")
        src = cv.imread("__tmp.png")
        # TODO: add x and y offest to the result rectangles
        ret = self.scanner.scan(src, bbox[0], bbox[1])
        return ret

class ImageScanner:

    def __init__(self, thresh : int = 100):
        self.thresh = thresh

    def updateThresh(self,thresh):
        self.thresh = thresh

    def scan(self,src : MatLike, x : int = 0, y : int = 0) -> None:
        # accept an input image and convert it to grayscale, and blur it
        gray_scale_image = grayscale_blur(src)

        # determine the bounding rectangles from canny detection
        _, bound_rect = canny_detection(gray_scale_image, min_threshold=self.thresh)

        # group the rectangles from this step
        grouped_rects = group_rects(bound_rect, 0, src.shape[1])

        grouped_rects = [(rect[0]+x,rect[1]+y,rect[2],rect[3]) for rect in grouped_rects]
        return grouped_rects
