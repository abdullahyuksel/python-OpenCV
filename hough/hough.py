# -*- coding: utf-8 -*-

"""
@ide:       SpyderEditor
@author:    Abdullah Yuksel
@project:   cevre.py
"""

import cv2
import numpy as np

img = cv2.imread("yol.jpg")

img_copy = img.copy()
gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 30, 50)


def nothing(x):
    pass

cv2.namedWindow("trackbar",cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar("threshold","trackbar",0,300,nothing)


while(1):
    img_copy = img.copy()
    
    threshold = cv2.getTrackbarPos("threshold", "trackbar")+1
    
    print(threshold)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold, 20, 0)
    
    if not isinstance(lines, type(None)):
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img_copy, (x1,y1), (x2,y2), (0,255,0),2)
    
    # lines = cv2.HoughLines(edges, 1, np.pi/180, threshold)
    
    # if not isinstance(lines, type(None)):
    #     for line in lines:
    #         for rho, theta in line:
    #             a = np.cos(theta)
    #             b = np.sin(theta)
    #             x0 = a*rho
    #             y0 = b*rho
                
    #             x1 = int(x0 + 1000*(-b))
    #             y1 = int(y0 + 1000*(a))
    #             x2 = int(x0 - 1000*(-b))
    #             y2 = int(y0 - 1000*(a))
                
    #             cv2.line(img_copy, (x1,y1), (x2,y2), (0,0,255), 2)
    
    cv2.imshow("trackbar",img_copy)
    
    if cv2.waitKey(33) & 0xFF == ord("q"):
        break
            

cv2.destroyAllWindows()