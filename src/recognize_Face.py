import cv2
import numpy as  np
import time
import sys
from keras.preprocessing import image
from keras.models import model_from_json
import pickle

model = model_from_json(open("../models/facial_expression_model_structure.json", "r").read())
model.load_weights('../models/facial_expression_model_weights.h5')

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

#Load Cascade Files
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #loading the face xml files
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') #xml for the eyes
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("TrainedFace.yml")
labels = {}
with open("face.labels", "rb") as f:
    init_labels = pickle.load(f)
    labels = {v:k for k,v in init_labels.items()} #reversing key value pairs inside dictionary
r = False
#img=cv2.imread('C:\\aaaaaaaa\\haarcascades\\ab.jpg')
cap=cv2.VideoCapture(0)
t1 = time.time();
while True:
    ret,img = cap.read() #Get each frame
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #converting to grayscale

    facial = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(35,35)
    )
    t2 = time.time();
    if r == False:
      dt1 = t2-t1
      print(dt1)
      r = True
    for (x,y,w,h) in facial:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2) #draw blue rectangle around eyes and lips
        roi_gray = gray[y:y+h, x:x+w] #locations of the face in grayscale for eyes
        roi_color = img[y:y+h, x:x+w] #locations of converted grayscale for eyes

        detected_face = img[int(y):int(y+h), int(x):int(x+w)]  #cropping the detected face area
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)  #converting to Gray
        detected_face = cv2.resize(detected_face, (48,48)) #Resizing the face

        #Predicting recognized face
        id_, confidence = recognizer.predict(detected_face) #predict


        if(confidence>120):
            print(id_, confidence)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,255,255)
            stroke = 2
            cv2.putText(img,name, (x+w, y+h), font,1, color, stroke, cv2.LINE_AA)

        #Predicting the Emotion
        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis = 0)        
        img_pixels /= 255        
        predictions = model.predict(img_pixels)

        max_index = np.argmax(predictions[0]) 
        emotion = emotions[max_index]
        cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor = 1.2,
            minNeighbors=5,
            minSize = (48,48)
        )# detect eyes
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew,ey+eh), (0,255,0),2) # drawing a blue rectangle around the eyes

    cv2.imshow('I know you so well',img) #Display the Output Stream

    #Exit Video Condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        sys.exit()

cap.release()
cv2.destroyAllWindows()