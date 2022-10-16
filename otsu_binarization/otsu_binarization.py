# -*- coding: utf-8 -*-

"""
@ide:       SpyderEditor
@author:    Abdullah Yuksel
@project:   otsubinarization.py
"""

import cv2
import matplotlib.pyplot as plt

resim = cv2.imread("shape_noise.png",0)

blur = cv2.GaussianBlur(resim,(15,15),0 )

ret, th2 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

print (ret)

plt.hist(resim.ravel(), 256)
plt.show()



cv2.imshow("resim",resim)
cv2.imshow("bl",blur)
cv2.imshow("th2",th2)
cv2.waitKey()
cv2.destroyAllWindows()
