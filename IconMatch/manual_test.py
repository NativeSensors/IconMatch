from PIL import ImageGrab
from pynput import mouse
import IconMatch
from IconMatch.helpers import run_sift, save_img


def main():
    """
    Manual test is an interactive screen shot tool that will perform SIFT on the image.
    todo: Add more options for edge detection algorithm and options on where to save image.
    todo: Add functionality for previewing before saving an image.
    """
    print("Choose the top left and bottom right coordinates of your box.")

    button_presses = []

    # Stream of events
    with mouse.Events() as events:
        for event in events:
            if hasattr(event, "button") and event.pressed:
                print(
                    "The {0} button was {1} at the following coordinates: x:{2}, y:{3}".format(
                        event.button,
                        "pressed" if event.pressed else "released",
                        event.x,
                        event.y,
                    )
                )
                button_presses.append(event)
            if len(button_presses) == 2:
                break

    screen_shot = ImageGrab.grab(
        bbox=(
            button_presses[0].x,
            button_presses[0].y,
            button_presses[1].x,
            button_presses[1].y,
        )
    )

    save_img(screen_shot, "pre")
    run_sift(screen_shot)
    save_img(screen_shot, "post")


if __name__ == "__main__":
    main()
