import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(True):
    # capture frame by frame
    ret, frame = cap.read()

    # display the resulting frame
    cv2.imshow("frame", frame)
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

# when everything is done, release the capture
cap.realease()
cv2.destroyAllWindows()
