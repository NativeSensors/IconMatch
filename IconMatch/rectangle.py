import math
import sys

from typing import List, Tuple


class Rectangle:
    """
    Representation of a rectangle in two dimensional space.

    Note: Keep in mind that due to OpenCV's representation of the screen, y increases from top to bottom!
                            0 - x +
                            |
                            y
                            +
    Where top, bottom, left and right are named relative to cartesian coordinates. The below
    diagram shows what this entails.
                           Top
                        --------
                        |      |
                   Left |      | Right
                        |      |
                        --------
                         Bottom
    """

    def __init__(self, top: int, left: int, bottom: int, right: int):
        """
        Create a rectangle in Cartesian notation
        """

        self.top = top
        self.left = left
        self.bottom = bottom
        self.right = right

        # stored for cached initialization
        self.area = None

    @staticmethod
    def intersect(rect_a: 'Rectangle', rect_b: 'Rectangle') -> bool:
        """
        Determine if rect_a, rect_b intersect.
        Modified slightly from:
        https://stackoverflow.com/questions/306316/determine-if-two-rectangles-overlap-each-other#306332
        """

        ret = (
                rect_a.left < rect_b.right
                and rect_a.right > rect_b.left
                and rect_a.top < rect_b.bottom
                and rect_a.bottom > rect_b.top
        )
        return ret

    @staticmethod
    def merge_rects(rects: List['Rectangle']) -> 'Rectangle':
        """
        Merge a list of rectangles into one conglomerate rect in Cartesian representation.
        """

        ans = Rectangle(sys.maxsize, sys.maxsize, 0, 0)

        for rect in rects:
            if rect.left < ans.left:
                ans.left = rect.left
            if rect.top < ans.top:
                ans.top = rect.top
            if rect.bottom > ans.bottom:
                ans.bottom = rect.bottom
            if rect.right > ans.right:
                ans.right = rect.right

        return ans

    @staticmethod
    def rect_cv_to_cartesian(rect: Tuple[int, int, int, int]) -> 'Rectangle':
        """
        Convert a rectangle from CV representation to cartesian coordinates.
        """
        new_rect = Rectangle(rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3])
        return new_rect

    @staticmethod
    def rect_cartesian_to_cv(rect: 'Rectangle') -> tuple:
        """
        Convert rectangle from Cartesian representation back to CV tuple representation
        """
        new_rect = (rect.top, rect.left, rect.bottom - rect.top, rect.right - rect.left)
        return new_rect

    def __eq__(self, other: 'Rectangle') -> bool:
        if isinstance(other, Rectangle):
            return (
                    self.top == other.top
                    and self.left == other.left
                    and self.right == other.right
                    and self.bottom == other.bottom
            )
        return False

    def __lt__(self, other: 'Rectangle') -> bool:
        # todo: Ideate a more absolute definition of "less than"
        if self.get_area() > other.get_area():
            return True
        return False

    def get_area(self) -> int:
        """
        Return the area taken up by this rectangle.
        """
        if self.area is None:
            self.area = (self.right - self.left) * (self.bottom - self.top)
        return self.area

    def contains_point(self, point: Tuple[int, int]) -> bool:
        """
        Given a Tuple representing a point, return whether the point is within this rectangle.
        """
        if self.distance_to_point(point) == 0.0:
            return True
        return False

    def distance_to_point(self, point: tuple) -> float:
        """
        Determine the distance from a rectangle to a point. If the point is within a rectangle, zero will be returned.
        https://stackoverflow.com/questions/5254838/calculating-distance-between-a-point-and-a-rectangular-box-nearest-point
        """
        dx = max(self.left - point[0], 0, point[0] - self.right)
        dy = max(self.top - point[1], 0, point[1] - self.bottom)

        if dx == 0 and dy == 0:
            return 0.0

        # potentially problematic due to using floats
        return math.sqrt(dx ** 2 + dy ** 2)

    def __str__(self):
        return "({1},{0}), ({3},{2})".format(self.top, self.left, self.bottom, self.right)
