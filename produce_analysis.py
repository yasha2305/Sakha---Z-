import cv2
import numpy as np

def estimate_produce_condition(image, name):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Count red pixels (ripe)
    red1 = cv2.inRange(hsv, (0,120,90), (10,255,255))
    red2 = cv2.inRange(hsv, (170,120,90), (180,255,255))
    red = red1 + red2

    # Count green pixels (unripe)
    green = cv2.inRange(hsv, (35,80,80), (85,255,255))

    red_ratio = np.sum(red > 0) / (image.size / 3)
    green_ratio = np.sum(green > 0) / (image.size / 3)

    if red_ratio > 0.4:
        return "Ripe", True, "Good to eat"
    elif green_ratio > 0.4:
        return "Unripe / Raw", False, "Let it ripen"
    else:
        return "Rotten", False, "Do not eat"