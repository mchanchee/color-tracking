"""
Detect an object of a given color and track it
"""


import cv2
import numpy as np

# Bounds in HSV for the range to detect
# Here we decide to detect blue
LOWER = np.array([105, 50, 50])
UPPER = np.array([135, 255, 255])

if __name__ == "__main__":

    camera = cv2.VideoCapture(0)
    tracked_points = []

    while cv2.waitKey(1) != 13: # 13 is the Enter key

        # Get frame, convert to HSV, extract contours
        ret, frame = camera.read()
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, LOWER, UPPER)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If we find no contour, go to next iteration
        if len(contours) == 0:
            continue

        # Take the biggest contour
        max_contour = max(contours, key=cv2.contourArea)
        center, radius = cv2.minEnclosingCircle(max_contour)

        # Convert to int so we can display them
        center = (int(center[0]), int(center[1]))
        radius = int(radius)

        # Display if the enclosing circle is large enough
        if radius > 30:
            cv2.circle(frame, center, int(radius), (0, 0, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            tracked_points.append(center)

            for i in range(1, len(tracked_points)):
                cv2.line(frame, tracked_points[i-1], tracked_points[i], (255, 255, 0), 2)
        else:
            # Empty trail
            tracked_points = []

        # Flip horizontally (around y-axis) and show
        frame = cv2.flip(frame, 1) 
        cv2.imshow("Colour tracking", frame)

camera.release()
cv2.destroyAllWindows()