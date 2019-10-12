import numpy as np
import cv2
from imageai.Prediction.Custom import CustomImagePrediction
import os
import webbrowser
import sys
import csv
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

#face_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv//4.0.1/share/opencv4/haarcascades/haarcascade_frontalface_alt2.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
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
prediction.setModelTypeAsDenseNet()
prediction.setModelPath(os.path.join(execution_path, "model_ex-038_acc-0.761834.h5"))
prediction.setJsonPath(os.path.join(execution_path, "model_class.json"))
prediction.loadModel(num_objects=5)

predictions, probabilities = prediction.predictImage(os.path.join(execution_path, "image.png"), result_count=5)
print("\n\n")
print("The model has the following outputs:\n\n")
#print(predictions)
#print(probabilities)
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction + " : " + eachProbability)

location_of_emo = probabilities.index(max(probabilities))
emo = predictions[location_of_emo]
print("\n\nYou seem to be {}.".format(emo))
media_input = int_check(1,3,"\nWhat is it that you'd like to be recommended based on your mood?\n\n1. Images\n2. Songs\n3. Movies\n\nPlease enter the corresponding digit: ")

if media_input==1: #Images
    if emo=='happy': #done
        webbrowser.open('https://www.buzzfeed.com/alexnaidus/laugh-therapy')
    elif emo=='sad': #done
        a = int_check(1,2,"\n\nWhat link would you like to view?\n1. 39 Pictures For Anyone Who Is Just Sad\n2. 42 Pictures That Will Make You Almost Too Happy\n\nPlease select a corresponding integer: ")
        if a==1:
            webbrowser.open('https://www.buzzfeed.com/kaelintully/bobby-flay-is-the-prince-of-sunshine-yes-yes-what-a-nice-boy')
        elif a==2:
            webbrowser.open('https://www.buzzfeed.com/kaelintully/someone-get-me-a-milano-bc-now-im-hungry')
    elif emo=='angry': #done
        a = int_check(1,2,"\n\nWhat link would you like to view?\n1. 21 Pictures That Will Definitely Make You Feel Better\n2. 28 Pictures That Will Help You Calm The Hell Down Today\n\nPlease select a corresponding integer: ")
        if a==1:
            webbrowser.open('https://www.buzzfeed.com/jessicamisener/it-gets-better')
        elif a==2:
            webbrowser.open('https://www.buzzfeed.com/daves4/pictures-that-will-help-you-get-over-your-intense')
    elif emo=='calm': #done
        a = int_check(1,2,"\n\nWhat link would you like to view?\n1. 24 Pictures For Anyone Who Needs A Little Visual Relaxation\n2. 16 Calming Websites That Will Put You At Ease\n\nPlease select a corresponding integer: ")
        if a==1:
            webbrowser.open('https://www.buzzfeed.com/jamiejones/sit-back-relax-and-let-these-pictures-soothe-your-soul')
        elif a==2:
            webbrowser.open('https://www.buzzfeed.com/anjalipatel/soothing-corners-of-the-internet-to-comfort-you-on-the-ba')
    elif emo=='surprised': #done
        webbrowser.open('https://www.buzzfeed.com/search?q=surprise')

elif media_input==2: #Songs
    if emo=='happy':
        webbrowser.get('open -a /Applications/Google\ Chrome.app %s').open('https://open.spotify.com/playlist/37i9dQZF1DX3rxVfibe1L0')
    elif emo=='sad':
        webbrowser.get('open -a /Applications/Google\ Chrome.app %s').open('https://open.spotify.com/playlist/37i9dQZF1DX3rxVfibe1L0')
    elif emo=='angry':
        webbrowser.get('open -a /Applications/Google\ Chrome.app %s').open('https://open.spotify.com/playlist/37i9dQZF1DX3rxVfibe1L0')
    elif emo=='calm':
        webbrowser.get('open -a /Applications/Google\ Chrome.app %s').open('https://open.spotify.com/playlist/37i9dQZF1DX6VdMW310YC7')
    elif emo=='surprised':
        webbrowser.get('open -a /Applications/Google\ Chrome.app %s').open('https://open.spotify.com/playlist/37i9dQZF1DX3rxVfibe1L0')

elif media_input==3: #Movies
    movie_keyword = input("\n\nWhat kind of genre do you like watching when you're {}?\nPlease type a keyword: ".format(emo))
    with open('movies.csv','a') as f:
        emowrite = csv.writer(f)
        emowrite.writerow([emo,movie_keyword])
    webbrowser.open("https://www.netflix.com/search?q={}".format(movie_keyword))
    
os.remove(os.path.join(execution_path,"image.png"))