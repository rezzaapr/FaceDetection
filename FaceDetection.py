import cv2
import numpy as np
import time
from urllib.request import urlopen

face_cascade = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('cascade/haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier('cascade/haarcascade_eye.xml')

# IP webcam 
url='http://192.168.43.1:8080/shot.jpg' 

while True:
    imgResp = urlopen(url) 
    imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
    img = cv2.imdecode(imgNp,-1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
       img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    smiles  = smile_cascade.detectMultiScale(img, scaleFactor = 1.8, minNeighbors = 20)
    for (sx, sy, sw, sh) in smiles:
       img = cv2.rectangle(img, (sx, sy), ((sx + sw), (sy + sh)), (0, 255,0), 2)
    eyes = eye_cascade.detectMultiScale(img, scaleFactor = 1.1, minNeighbors = 5)
    for (ex,ey,ew,eh) in eyes:
       img = cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
       break
    cv2.imshow("Smile Detected", img)
