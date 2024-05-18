import argparse
import random as rng
import cv2 as cv

from icondetection.box import grayscale_blur, canny_detection, group_rects, candidate_rectangle
from icondetection.rectangle import Rectangle


def closest_rectangle_handler(event: int, x: int, y: int, flags, params) -> None:
    """
    Determine the closest rectangle to mouse click.
    https://divyanshushekhar.com/mouse-events-opencv/
    TODO: The ordering of x and y appears to be inconsistent. Check highlighted line
    """

    # globals necessary for access in callback function
    global src, src2, candidate_rect, grouped_rects, excluded_rects

    if event == cv.EVENT_LBUTTONDOWN:
        print("x coordinate:{}, y coordinate: {}".format(x, y))
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))

        src2 = src.copy()
        candidate_rect = candidate_rectangle([Rectangle.rect_cv_to_cartesian(t) for t in grouped_rects], (y, x))  # TODO
        excluded_rects = filter(lambda rect: rect is not candidate_rect, grouped_rects)

        cv.rectangle(
            src2,
            (candidate_rect.bottom, candidate_rect.left),
            (candidate_rect.top, candidate_rect.right),
            color,
            2,
        )
        cv.imshow("Candidate Rectangles", src2)


def null_handler(event: int, x: int, y: int, flags, params) -> None:
    """A placeholder handler that does nothing."""
    pass


def candidate_rectangle_demo() -> None:
    """Show a candidate rectangle for a pressed location"""

    cv.imshow("Candidate Rectangles", src2)
    cv.setMouseCallback("Candidate Rectangles", closest_rectangle_handler)


def render_rectangles(rectangles, input_image, display_text, callback=null_handler,
                      desired_color: tuple = None) -> None:
    """
    Render given rectangles on provided input image.
    Note: Make sure to send a copy of your image with .copy()
    """

    # TODO: may not need to have specialized conversion from different rect
    #       types
    for index in range(len(rectangles)):
        color = desired_color if desired_color is not None else (
            rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256)
        )
        cv.rectangle(
            input_image,
            (int(rectangles[index][0]), int(rectangles[index][1])),
            (
                int(rectangles[index][0] + rectangles[index][2]),
                int(rectangles[index][1] + rectangles[index][3]),
            ),
            color,
            2,
        )

    cv.imshow(display_text, input_image)
    cv.setMouseCallback(display_text, callback)


def threshold_callback(val: int) -> None:
    """
    Takes a value of threshold for the canny edge detector and finds the bounding rectangles of appropriate edges
    within an image.
    """

    # accept an input image and convert it to grayscale, and blur it
    gray_scale_image = grayscale_blur(src)

    # determine the bounding rectangles from canny detection
    _, bound_rect = canny_detection(gray_scale_image, min_threshold=val)

    # group the rectangles from this step
    global grouped_rects
    grouped_rects = group_rects(bound_rect, 0, src.shape[1])

    # (for display purposes) use the provided rectangles to display in your program
    render_rectangles(grouped_rects, src.copy(), "Grouped Rectangles", desired_color=(36, 9, 14))
    render_rectangles(bound_rect, src.copy(), "Original Rectangles", desired_color=(96, 9, 104))
    candidate_rectangle_demo()


if __name__ == "__main__":
    rng.seed(12345)
    parser = argparse.ArgumentParser(
        description="Sample showcase of IconDetection."
    )

    parser.add_argument("--input", help="Path to input image.")

    args = parser.parse_args()
    src = cv.imread(args.input)
    src2 = cv.imread(args.input)  # image for closest rectangle detection

    if src is None:
        print("Could not open or find the image:", args.input)
        exit(0)

    src_gray = grayscale_blur(src)

    source_window = "Source"
    cv.namedWindow(source_window)
    cv.imshow(source_window, src)

    max_thresh = 255
    thresh = 100  # initial threshold

    cv.createTrackbar(
        "Canny threshold:", source_window, thresh, max_thresh, threshold_callback
    )
    threshold_callback(thresh)
    cv.waitKey()
