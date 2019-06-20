import cv2
import os
from PIL import Image
import numpy as np
import pickle
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #loading the face xml files
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) #fetching the directory of this file
image_dir = os.path.join(BASE_DIR, "images") #getting the images directory path
recognizer = cv2.face.LBPHFaceRecognizer_create()

x_train = []     #img data for training
y_labels = []     #fetch individual labels and store it here for label
current_id = 0
label_ids = {}
for root, dirs, files in os.walk(image_dir):  #walk inside images directory and each files
    for file in files:
        if file.endswith("jpg") or file.endswith("png"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(" ", "_").lower() #fetching folder names and cleaning data
            print(label, path)
            ##TO-DO make lables into numbers
            ##TO-DO verify and convert image into Grayscale and into numpy array
            pil_image = Image.open(path).convert("L") #converting image to grayscale
            image_array = np.array(pil_image, "uint8")
            if not label in label_ids:
                label_ids[label] = current_id
                current_id+=1
            id_ = label_ids[label]
            print(label_ids)
            faces = face_cascade.detectMultiScale(
                image_array,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(35,35)
            )
            for x,y,w,h in faces:
                roi = image_array[int(y):int(y+h), int(x):int(x+w)]  #cropping the detected face area
                x_train.append(roi)
                y_labels.append(id_)

print(x_train, y_labels)
with open("face.labels", "wb") as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("TrainedFace.yml")
print("\n\n**********I'm educated! Let's start recognizing people.**********\n\n")








