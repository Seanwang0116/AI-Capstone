import cv2
import numpy as np

stop_training = False
def stop_callback(event, x, y, flags, param):
    global stop_training
    if event == cv2.EVENT_LBUTTONDOWN:
        stop_training = True

cv2.namedWindow("Control")
cv2.setMouseCallback("Control", stop_callback)
control_image = np.zeros((100, 300, 4), dtype=np.uint8)
cv2.putText(control_image, "Click here to STOP training", (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
cv2.imshow("Control", control_image)
cv2.waitKey(1)