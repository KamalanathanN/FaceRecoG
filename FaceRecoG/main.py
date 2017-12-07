import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);


cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
while (True):
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        id,conf = recognizer.predict(gray[y:y+h,x:x+w])
        if(id==1):
            id="jon snow" #Enter name of id 1
        elif(id==2):
            id="sam" #Enter name of id 2
        else:
            id="new person"
        cv2.putText(im,str(id),(x,y+h+50), font,1,(1,255,1),2)
    cv2.imshow('im',im) 
    if (cv2.waitKey(1)==ord('q')):
        break;
cam.release()
cv2.destroyAllWindows()
