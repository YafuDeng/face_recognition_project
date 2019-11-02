import cv2
from cv2 import face
import numpy as np
import pickle

face_cascades = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")
recognizer = face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

# give label from the pkl file we have
labels = {"person_name": 1}
with open("labels.pkl", "rb") as f:
    og_labels = pickle.load(f)
    # reverse the label number to name
    labels = {v:k for k,v in og_labels.items()}



cap = cv2.VideoCapture(0)

while(True):
    # capture frame by frame
    ret, frame = cap.read()

    # since the cascade only works in gray color
    # we need to convert the frames to gray coloe
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # find all faces in this frame
    # scaleFactor and minNeighbors can be changed to other numbers
    # these number are straight from the documentation
    # might help to have better result if changed
    faces = face_cascades.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)

    # iterate through these faces
    # (x,y,w,h) means the region of interests
    for(x,y,w,h) in faces:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_coloe = frame[y:y+h, x:x+w]

        # recoglize? deep learning model
        # give label back and the confidence
        # can be modified to do better
        # conf value needs some research
        id_, conf = recognizer.predict(roi_gray)
        if conf>= 45 and conf <=85:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (225, 225, 225)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

        # this can be used to store a lot of images for that person
        # which can be used to expand the learning pool
        img_item = "my-image.png"
        cv2.imwrite(img_item, roi_gray)

        # draw a rectangle
        color = (255, 0, 0) # it's in BGR
        stroke = 2
        end_x = x+w
        end_y = y+h
        cv2.rectangle(frame, (x,y), (end_x, end_y), color, stroke)

    # display the resulting frame
    cv2.imshow("frame", frame)
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

# when everything is done, release the capture
cap.realease()
cv2.destroyAllWindows()
