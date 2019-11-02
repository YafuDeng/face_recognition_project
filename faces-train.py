import os
import cv2
from cv2 import face
import pickle
import numpy as np
from PIL import Image

# whereever the file is saved
# looking for the path of it
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# path of the images
image_dir = os.path.join(BASE_DIR, "images")

face_cascades = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")
# face recognizer:
recognizer = face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []


for root, dir, files in os.walk(image_dir):
    for file in files:
        if file.endswith("jpg") or file.endswith("png"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
            #print(label, path)
            # add the label into label_ids dic if it's not there already
            # then plus current_id by 1
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1

            id_ = label_ids[label]
            #print(label_ids)
            # need some bumber to represent labels,
            # verufy the image and turn it into a numpy array, GRAY
            pil_image = Image.open(path).convert("L") # grayscale

            # resize the image for better trainning result
            size = (550, 550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)


            image_array = np.array(pil_image, "uint8") # turning the image into numpy array
            #print(image_array)
            faces = face_cascades.detectMultiScale(image_array, scaleFactor = 1.5, minNeighbors = 5)

            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)



#print("y_labels ", y_labels)
#print("x_train",  x_train)

with open("labels.pkl", "wb") as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")
