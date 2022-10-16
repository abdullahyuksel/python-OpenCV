# -*- coding: utf-8 -*-
"""
@ide:       SpyderEditor
@author:    Abdullah Yuksel
@project:   pelerin.py
"""
import cv2
import numpy as np

cam = cv2.VideoCapture(1)

lower = np.array([0,144,84])
upper = np.array([14,255,255])


kernel = np.ones((3,3),np.uint8)
kernel2 = np.ones((11,11),np.uint8)
kernel3 = np.ones((13,13),np.uint8)

_, background = cam.read()

while(cam.isOpened()):
    
    _, frame = cam.read()
    
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    
    mask = cv2.inRange(hsv,lower,upper)
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel2)
    mask = cv2.dilate(mask,kernel3,iterations=2)
    
    
    mask_not = cv2.bitwise_not(mask)
    
    bg = cv2.bitwise_and(background,background,mask=mask)
    fg = cv2.bitwise_and(frame,frame,mask=mask_not)
    
    dst = cv2.addWeighted(bg,1,fg,1,0)


    # dst = np.vstack((frame,dst))

    cv2.imshow("orijinal", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("dst", dst)
    
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
cam.release()
cv2.destroyAllWindows()