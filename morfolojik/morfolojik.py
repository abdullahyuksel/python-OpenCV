# -*- coding: utf-8 -*-

"""
@ide:       SpyderEditor
@author:    Abdullah Yuksel
@project:   morfolojik.py
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

resim = cv2.imread("D:/gitHub/openCV/morfolojik/1-1i.png",0)

kernel = np.ones((6,6),np.uint8)

erosion = cv2.erode(resim,kernel, iterations=1)
dilation = cv2.dilate(resim,kernel, iterations=1)
opennig = cv2.morphologyEx(resim, cv2.MORPH_OPEN, kernel, iterations=1)
closing = cv2.morphologyEx(resim, cv2.MORPH_CLOSE, kernel, iterations=1)
tophat = cv2.morphologyEx(resim, cv2.MORPH_TOPHAT, kernel, iterations=1)
blackhat = cv2.morphologyEx(resim, cv2.MORPH_BLACKHAT, kernel, iterations=1)
gradient = cv2.morphologyEx(resim, cv2.MORPH_GRADIENT, kernel, iterations=1)


plt.subplot(241),plt.imshow(cv2.cvtColor(resim, cv2.COLOR_BGR2RGB)),plt.title("original")
plt.subplot(242),plt.imshow(cv2.cvtColor(erosion, cv2.COLOR_BGR2RGB)),plt.title("erosion")
plt.subplot(243),plt.imshow(cv2.cvtColor(dilation, cv2.COLOR_BGR2RGB)),plt.title("dilation")
plt.subplot(244),plt.imshow(cv2.cvtColor(opennig, cv2.COLOR_BGR2RGB)),plt.title("opennig")
plt.subplot(245),plt.imshow(cv2.cvtColor(closing, cv2.COLOR_BGR2RGB)),plt.title("closing")
plt.subplot(246),plt.imshow(cv2.cvtColor(tophat, cv2.COLOR_BGR2RGB)),plt.title("tophat")
plt.subplot(247),plt.imshow(cv2.cvtColor(blackhat, cv2.COLOR_BGR2RGB)),plt.title("blackhat")
plt.subplot(248),plt.imshow(cv2.cvtColor(gradient, cv2.COLOR_BGR2RGB)),plt.title("gradient")

plt.show()

cv2.imshow("resim",resim)
cv2.imshow("erosion",erosion)
cv2.imshow("dilation",dilation)



cv2.waitKey()
cv2.destroyAllWindows()