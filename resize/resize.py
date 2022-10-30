# -*- coding: utf-8 -*-

import cv2

img1 = cv2.imread("1.jpg")
print(img1.shape)

r = cv2.resize(img1, (1000,500))

cv2.imshow("resize", r)
cv2.imshow("orj", img1)

cv2.waitKey(0)
cv2.destroyAllWindows()