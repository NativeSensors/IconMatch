import argparse
import random as rng
import cv2 as cv

from PIL import ImageGrab
from cv2.typing import MatLike
from icondetection.box import grayscale_blur, canny_detection, group_rects, candidate_rectangle
from icondetection.rectangle import Rectangle

class ScreenScanner:

    def __init__(self):

        self.scanner = IconScanner()
        self.thresh = 100
        pass

    def scan(self):
        screenshot = ImageGrab.grab()
        screenshot.save("__tmp.png")
        src = cv.imread("__tmp.png")
        ret = self.scanner.scan(src, self.thresh)
        return ret
 
class IconScanner:

    def __init__(self):
        pass

    def scan(self,src : MatLike,val: int) -> None:
        # accept an input image and convert it to grayscale, and blur it
        gray_scale_image = grayscale_blur(src)

        # determine the bounding rectangles from canny detection
        _, bound_rect = canny_detection(gray_scale_image, min_threshold=val)

        # group the rectangles from this step
        grouped_rects = group_rects(bound_rect, 0, src.shape[1])
        return grouped_rects
    
if __name__ == "__main__":
    scanner = ScreenScanner()
    ret = scanner.scan()
    print(len(ret),ret[0])
    