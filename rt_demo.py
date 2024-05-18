import argparse
import cv2 as cv
import numpy as np
import random as rng
import queue
import time 

import sys
from PySide2.QtCore import Qt, QTimer
from PySide2.QtGui import QPainter, QColor, QBrush, QPen
from PySide2.QtWidgets import QApplication, QMainWindow, QWidget

from pynput import mouse
from PIL import ImageGrab
from cv2.typing import MatLike
from icondetection.box import grayscale_blur, canny_detection, group_rects, candidate_rectangle
from icondetection.rectangle import Rectangle


class CircleWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.diameter = 50
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.WindowTransparentForInput)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setGeometry(0, 0, self.diameter+10, self.diameter+10)

        self.setColor(204, 54, 54)

        self.to_y = self.x()
        self.to_x = self.y()


        self.pos_call = None

        # Start a timer to update the position periodically
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_position)
        self.timer.start(30)  # Update every second

    def connect_pos_call(self,pos_call):
        self.pos_call = pos_call

    def update_position(self):
        # Randomly generate new position within the screen boundaries
        if self.pos_call:
            x,y = self.pos_call()
            self.setPosition(x,y)

        new_x = self.to_x
        new_y = self.to_y

        if self.geometry().width() != self.diameter + 10:
            self.setGeometry(0, 0, self.diameter + 10, self.diameter + 10)
        self.move(new_x, new_y)
        self.repaint()

    def setPosition(self,x,y):
        self.to_x = x - self.diameter/2
        self.to_y = y - self.diameter/2

    def setRadius(self,diameter):
        self.diameter = diameter

    def setColor(self,r,g,b):
        self.brush_color = QColor(r,g,b, 50)
        self.pen_color = QColor(r, g, b)  # Red color for the border

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # brush_color = QColor(18, 34, 78, 50)  # Semi-transparent red color
        brush = QBrush(self.brush_color)
        painter.setBrush(brush)
        # Set the pen color and width
        pen_width = 2  # Width of the border
        pen = QPen(self.pen_color, pen_width)
        painter.setPen(pen)

        # Draw a circle
        painter.drawEllipse((self.width() - self.diameter) / 2, (self.height() - self.diameter) / 2, self.diameter, self.diameter)

class CursorTracker:

    def __init__(self,rate=5):
        window_h = 3000
        window_w = 3000
        self.scanner = ScreenScanner()
        rectangles = self.scanner.scan(bbox = (0,0,window_w,window_h))

        self.DSB = DynamicSpatialBuckets()
        self.DSB.loadData(rectangles)
        self.mouse = mouse.Controller()

        self.start = time.time()
        self.rate = rate

    def rescan(self,x,y,w,h):
        print(x,y,w,h)
        rectangles = self.scanner.scan(bbox = (x,y,w,h))
        self.DSB = DynamicSpatialBuckets()
        self.DSB.loadData(rectangles)

    def getPos(self):

        x,y = self.mouse.position
        print(x,y)
        if((time.time() - self.start) > self.rate):
            self.start = time.time()
            self.rescan(x,y,x+3000,y+3000)

        rectangles = self.DSB.getBucket([x,y])
        closest_distance = 9999
        closest_rectangle = None

        for rectangle in rectangles:
            center_x = rectangle[0] + rectangle[2]/2
            center_y = rectangle[1] + rectangle[3]/2
            distance = np.linalg.norm(np.array([center_x,center_y]) - np.array([x,y]))
            if distance < closest_distance:
                closest_distance = distance
                closest_rectangle = rectangle

        if closest_rectangle:
            return (closest_rectangle[0],closest_rectangle[1])
        return 0,0

class DynamicSpatialBuckets:

    def __init__(self):

        self.buckets = [[]]
        self.step = 500

    def loadData(self,rectangles):

        for rectangle in rectangles:
            center_x = rectangle[0] + rectangle[2]/2
            center_y = rectangle[1] + rectangle[3]/2

            index_x = int(center_x/self.step)
            index_y = int(center_y/self.step)

            while(index_x >= len(self.buckets)):
                self.buckets.append([])
            
            while(index_y >= len(self.buckets[index_x])):
                self.buckets[index_x].append([])

            self.buckets[index_x][index_y].append(rectangle)

    def getBucket(self,point):
        
        index_x = int(point[0]/self.step)
        index_y = int(point[1]/self.step)

        ret_bucket = []
        if len(self.buckets) > index_x and len(self.buckets[index_x]) > index_y:
            ret_bucket = self.buckets[index_x][index_y]

        return ret_bucket
class ScreenScanner:

    def __init__(self):

        self.scanner = IconScanner()
        self.thresh = 100
        pass

    def scan(self,bbox = None):
        screenshot = ImageGrab.grab(bbox = bbox)
        screenshot.save("__tmp.png")
        src = cv.imread("__tmp.png")
        # TODO: add x and y offest to the result rectangles
        ret = self.scanner.scan(src, self.thresh, bbox[0], bbox[1])
        return ret
class IconScanner:

    def __init__(self):
        pass

    def scan(self,src : MatLike,val: int, x : int = 0, y : int = 0) -> None:
        # accept an input image and convert it to grayscale, and blur it
        gray_scale_image = grayscale_blur(src)

        # determine the bounding rectangles from canny detection
        _, bound_rect = canny_detection(gray_scale_image, min_threshold=val)

        # group the rectangles from this step
        grouped_rects = group_rects(bound_rect, 0, src.shape[1])

        grouped_rects = [(rect[0]+x,rect[1]+y,rect[2],rect[3]) for rect in grouped_rects]
        return grouped_rects
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    dot = CircleWidget()
    dot.show()
    tracker = CursorTracker()
    dot.connect_pos_call(tracker.getPos)
    sys.exit(app.exec_())
    