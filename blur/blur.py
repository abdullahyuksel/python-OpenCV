# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
@ide:       SpyderEditor
@author:    Abdullah Yuksel
@project:   blur.py
"""

import cv2
import numpy as np


img = cv2.imread("1.jpg")

# kernel = np.ones((10,10),np.float32)/100
# dst = cv2.filter2D(img,-1,kernel)

# dst = cv2.blur(img,(10,10))


# dst = cv2.GaussianBlur(img,(7,7),0)

# dst = cv2.medianBlur(img, 7)

# filter = np.array([[-1,-1,-1],
#                    [-1,9,-1],
#                    [-1,-1,-1]])

# filter = np.array([[0,0,-1, 0,0],
#                    [0, -1,-2,-1,0],
#                    [-1,-2,16,-2,-1],
#                    [0,-1,-2,-1,0],
#                    [0,0,-1,0,0]])


# filter = np.array([[4,-3,-4],
#                    [-5,7,-7],
#                    [3,-2,-1]])

# dst = cv2.filter2D(img,-1,filter)


# filter = np.array([[0.272, 0.534, 0.131],
#                    [0.349, 0.686, 0.168],
#                    [0.393, 0.769, 0.189]])

# dst = cv2.transform(img, filter)


filter = np.array([[0,1,0],
                   [0,0,0],
                   [0,-1,0]])

dst = cv2.filter2D(img,-1,filter) + 64

cv2.imshow("img",img)
cv2.imshow("dst",dst)
cv2.waitKey()

cv2.destroyAllWindows()