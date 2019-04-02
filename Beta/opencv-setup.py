import numpy as np
import cv2

cap = cv2.VideoCapture(0)

def make_1080p(): #defining resolution to 1080p
    cap.set(3, 1920)
    cap.set(4, 1080)

def make_720p(): #defining resolution to 720p
    cap.set(3, 1280)
    cap.set(4, 720)

def make_480p(): #defining resolution to 480p
    cap.set(3, 640)
    cap.set(4, 480)

def change_res(width, height): #defining resolution to custom
    cap.set(3, width)
    cap.set(4, height)

make_480p()

while(True):
    # Capture frame-by-frame
    ret,frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    # Display the resulting frame
    cv2.imshow('color input through webcam',frame)
    cv2.imshow('gray',gray)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()