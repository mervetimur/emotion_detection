import cv2
import numpy as np

face_cascade=cv2.CascadeClassifier('C:/Users/Hp/Desktop/python.py/kameraacma_yuztanima.py/haarcascade_frontalface_default.xml')

cap=cv2.VideoCapture(0) 

while True:
    ret, frame=cap.read()
    grey=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(grey,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    

    cv2.imshow("video", frame)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()