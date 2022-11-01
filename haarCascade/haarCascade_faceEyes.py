# -*- coding: utf-8 -*-

"""
@ide:       SpyderEditor
@author:    Abdullah Yuksel
@project:   haarCascade_faceEyes.py
"""

import cv2
import time

start_loop = time.time()

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

img = cv2.imread("abdullah.PNG")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(img_gray, 1.3, 10)

for x, y, w, h in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 3)
    
    roi_gray = img_gray[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 10)
    
    for ex, ey, ew, eh in eyes:
        cv2.rectangle(img, (ex+x, ey+y), (ex+ew+x, ey+eh+y), (0,255,0), 2)
                      
stop_loop = time.time()

total_time = stop_loop - start_loop

print (total_time)
cv2.imshow("img_gray", img_gray)        
cv2.imshow("img", img)
cv2.waitKey()
cv2.destroyAllWindows()
    
    