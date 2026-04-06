import cv2
import numpy as np

cap = cv2.VideoCapture(0)

def detect_object(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])

    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = red_mask1 + red_mask2

    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    red_pixels = np.sum(red_mask > 0)
    green_pixels = np.sum(green_mask > 0)

    if red_pixels > 5000:
        return "Red Chili (Ripe)"
    elif green_pixels > 5000:
        return "Green Chili (Raw)"
    else:
        return "Other Object"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    result = detect_object(frame)

    cv2.putText(frame, result, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    cv2.imshow("Object Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()