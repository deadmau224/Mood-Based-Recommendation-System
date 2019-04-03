import numpy as np
import cv2
from imageai.Prediction.Custom import CustomImagePrediction
import os

def make_720p(): #defining resolution to 720p
    cap.set(3, 1280)
    cap.set(4, 720)

def int_check(l,h,message): #for inputting integers
    while True:
        s = input(message)
        if s.isdigit():
            s=int(s)
            if l<=s<=h:
                return s
            else:
                print("\nError in input, please try again.")
        else:
            print("\nError in input, please try again.")

#Capturing Face

face_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv//4.0.1/share/opencv4/haarcascades/haarcascade_frontalface_alt2.xml')
cap = cv2.VideoCapture(0)
make_720p()
while(True): #Capture frame-by-frame
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors = 5)
    for(x,y,w,h) in faces:
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]
        color = (255,0,0)
        stroke = 2
        cv2.rectangle(frame,(x,y),(x+w,y+h), color, stroke)
        cv2.imwrite("image.png", roi_gray)
   
    cv2.imshow('Live Webcam',frame) #Display the resulting frame
    if cv2.waitKey(1) & 0xFF == ord('z'):
        break
cap.release()
cv2.destroyAllWindows()

#Detecting Emotion using Machine Learning model that I trained for about 4 hours. Accuracy of model is 68.28%.

execution_path = os.getcwd() #Get current directory path

prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath(os.path.join(execution_path, "model_ex-020_acc-0.732311.h5"))
prediction.setJsonPath(os.path.join(execution_path, "model_class.json"))
prediction.loadModel(num_objects=5)

predictions, probabilities = prediction.predictImage(os.path.join(execution_path, "image.png"), result_count=5)

location_of_emo = probabilities.index(max(probabilities))
emo = predictions[location_of_emo]
print("\n\nYou are looking",predictions[location_of_emo],".")
media_input = int_check(1,3,"\nWhat is it that you'd like to be recommended based on your mood?\n\n1. Images\n2. Songs\n3. Movies\n\nPlease enter the corresponding digit: ")

if media_input==1: #Images
    if emo=='happy':
        print("Yet to code.")
    elif emo=='sad':
        print("Yet to code.")
    elif emo=='angry':
        print("Yet to code.")
    elif emo=='calm':
        print("Yet to code.")
    elif emo=='surprised':
        print("Yet to code.")
elif media_input==2: #Songs
    if emo=='happy':
        print("Yet to code.")
    elif emo=='sad':
        print("Yet to code.")
    elif emo=='angry':
        print("Yet to code.")
    elif emo=='calm':
        print("Yet to code.")
    elif emo=='surprised':
        print("Yet to code.")
elif media_input==3: #Movies
    if emo=='happy':
        print("Yet to code.")
    elif emo=='sad':
        print("Yet to code.")
    elif emo=='angry':
        print("Yet to code.")
    elif emo=='calm':
        print("Yet to code.")
    elif emo=='surprised':
        print("Yet to code.")