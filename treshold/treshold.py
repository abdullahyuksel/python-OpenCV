# -*- coding: utf-8 -*-

"""
@ide:       SpyderEditor
@author:    Abdullah Yuksel
@project:   threshold.py
"""

import cv2
import th_func

resim = cv2.imread ("gradient.jpg",0)

"""
ret, resim_thresh = cv2.threshold(resim,182,255,cv2.THRESH_BINARY)
"""
ret, resim_thresh = th_func.threshold(resim, 182, 255)

cv2.imshow("resim2",resim_thresh)
cv2.imshow("resim",resim)
cv2.waitKey()
cv2.destroyAllWindows()