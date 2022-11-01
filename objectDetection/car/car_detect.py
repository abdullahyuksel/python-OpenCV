# -*- coding: utf-8 -*-
"""
@ide:       SpyderEditor
@author:    Abdullah Yuksel
@project:   car_detection.py
"""

import cv2
import time


car_cascade = cv2.CascadeClassifier("cars.xml")
cam = cv2.VideoCapture("car1.avi")

while cam.isOpened():
    start_loop = time.time()
    ret, frame = cam.read()
    
    if not ret:
        print("done")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    gray = cv2.equalizeHist(gray)
    
    cars = car_cascade.detectMultiScale(gray, 1.1, 3)
    
    for x, y, w, h in cars:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        #frame[y:y+h, x:x+w, 0] = 255
    
    cv2.imshow("frame",frame)
    stop_loop = time.time()

    total_time = stop_loop - start_loop
    print (total_time)
    
    if cv2.waitKey(33) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
