#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
@ide:       SpyderEditor
@author:    Abdullah Yuksel
@project:   nesne_takip.py
"""

import cv2
import numpy as np
import RPi.GPIO as GPIO
from time import sleep

pinMode = 12
GPIO.setmode(GPIO.BOARD)
GPIO.setup(pinMode, GPIO.OUT)
pwm=GPIO.PWM(pinMode, 50)
pwm.start(0)

def SetAngle(angle):
    duty = angle / 18 + 2
    pwm.ChangeDutyCycle(duty)
    sleep(0.1)
    pwm.ChangeDutyCycle(0)
    

cap = cv2.VideoCapture(0)

# Set camera resolution
cap.set(3, 320.0)
cap.set(4, 240.0)
rows, cols = cap.get(4), cap.get(3)

x_medium = int(cols / 2)
center = int(cols / 2)
position = 90 # degrees
# sapma = rows/10
sapma = 30


def filt(img, low=np.array([161, 155, 84]), high=np.array([179, 255, 255])):
	hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	hsv_frame = cv2.inRange(hsv_frame, low, red)
	contours, _ = cv2.findContours(hsv_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x:cv2.contourArea(x), reverse=True)
    return contours

try:
    while cap.isOpened():
        _, frame = cap.read()
        
        contours = filt(frame)
        
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            x_medium = int(x + w / 2)
            break
        
        # Move servo motor
        if x_medium < center - sapma:
            if position < 180:
                position += 1
                SetAngle(position)
        elif x_medium > center + sapma:
            if position > 0:
                position -= 1
                SetAngle(position)
            
        print(position)
        
        cv2.line(frame, (x_medium, 0), (x_medium, 300), (0, 255, 0), 2)
        
        cv2.imshow("Frame", frame)
        
        if cv2.waitKey(33) == 27:
            break

except Exception as e:
    print(e)
    pass

cap.release()
cv2.destroyAllWindows()  
pwm.stop()
GPIO.cleanup()