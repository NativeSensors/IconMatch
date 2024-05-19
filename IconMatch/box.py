import heapq

import cv2 as cv
from typing import List

from IconMatch.rectangle import Rectangle
from IconMatch.weighted_quick_unionUF import WeightedQuickUnionUF as uf


def containing_rectangle(rects: List[Rectangle], query_point: tuple) -> Rectangle or None:
    """
    Provide the rectangle that covers this query point. Return None if there is no overlap.
    TODO: Currently non-deterministic due to iterating through an unordered list.
    """

    # brute force implementation for now
    for rect in rects:
        if rect.contains_point(query_point):
            return rect

    return None


def closest_rectangle(rects: List[Rectangle], query_point: tuple) -> Rectangle:
    """
    Determine the closest rectangle to this query point.
    TODO: Currently non-deterministic due to iterating through an unordered list.
    """

    closest_distance = rects[0].distance_to_point(query_point)
    closest_rect = rects[0]

    for rect in rects[1:]:
        temp_dist = rect.distance_to_point(query_point)
        if temp_dist < closest_distance:
            closest_distance = temp_dist
            closest_rect = rect

    return closest_rect


def candidate_rectangle(rects: List[Rectangle], query_point: tuple) -> Rectangle:
    """
    Return the closest rectangle, when given a query point in two dimensional space and a list of rectangles.
    todo: Correctness is not yet guaranteed.
    """

    # first verify if the query_point is within a rectangle. If it is, return this rectangle.
    potential_rect = containing_rectangle(rects, query_point)
    if potential_rect is not None:
        return potential_rect

    # we now check what is the closest individual point to the query_point.
    potential_rect = closest_rectangle(rects, query_point)

    return potential_rect


def grayscale_blur(image):
    """
    Convert image to gray and blur it.
    """
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_gray = cv.blur(image_gray, (3, 3))

    return image_gray


def rect_list_to_dict(rects):
    """
    Take a list of cv rects and returns their dictionary representation for simple filtering.
    """
    # POTENTIALLY PROBLEMATIC: necessarily has to be a list of (index, rects)
    rect_dict = {}
    rect_list = [None] * len(rects)

    for rect_index in range(len(rects)):
        temp_rect = Rectangle.rect_cv_to_cartesian(rects[rect_index])

        # step to modify dictionary
        if temp_rect.left in rect_dict:
            tmp_list = rect_dict[temp_rect.left]
            tmp_list.append((rect_index, temp_rect))
            rect_dict[temp_rect.left] = tmp_list
        else:
            rect_dict[temp_rect.left] = [(rect_index, temp_rect)]

        # step to modify list
        rect_list[rect_index] = temp_rect

    return rect_dict, rect_list


def group_rects(cv_rects, min_x, max_x):
    """
    Accepts a list of rects in openCV format, and groups them according to their
    overlapping locations.
    """

    rect_dict, rect_list = rect_list_to_dict(cv_rects)
    rect_heap = []
    unified_rects = uf(len(rect_list), rect_list)

    for x in range(min_x, max_x):

        # prune any outdated rects from the current_rects
        while True:
            if len(rect_heap) == 0:
                break
            if rect_heap[0][0] == x - 1:  # means we are at the edge
                heapq.heappop(rect_heap)
            else:
                break

        # get rectangles in this index
        if x in rect_dict:
            temp_rects = rect_dict[x]
        else:
            continue

        # perform intersection
        for rectA in rect_heap:
            for rectB in temp_rects:
                if Rectangle.intersect(rectA[1], rectB[1]):
                    unified_rects.union(rectA[2], rectB[0])

        # push new elements onto heap
        for rectB in temp_rects:
            heapq.heappush(rect_heap, (rectB[1].right, rectB[1], rectB[0]))

    # perform groupings
    grouped_rects = []
    unions = unified_rects.get_unions()
    for group in unions.values():
        grouped_rects.append(
            Rectangle.rect_cartesian_to_cv(Rectangle.merge_rects(group))
        )

    return grouped_rects


def canny_detection(gray_scale_image=None, **kwargs):
    """
    Run openCV Canny detection on a provided gray scale image. Return the polygons of canny contours and bounding
    rectangles.
    https://docs.opencv.org/3.4/da/d0c/tutorial_bounding_rects_circles.html
    """

    multiplier = kwargs['multiplier'] if 'multiplier' in kwargs else 2
    contour_accuracy = kwargs['contour_accuracy'] if 'multiplier' in kwargs else 3
    min_threshold = kwargs['min_threshold'] if 'min_threshold' in kwargs else 100
    max_threshold = int(min_threshold * multiplier)

    canny_output = cv.Canny(gray_scale_image, min_threshold, max_threshold)

    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contours_poly = [None] * len(contours)
    bound_rect = [None] * len(contours)

    for index, contour in enumerate(contours):
        contours_poly[index] = cv.approxPolyDP(contour, contour_accuracy, True)
        bound_rect[index] = cv.boundingRect(contours_poly[index])

    return contours_poly, bound_rect
