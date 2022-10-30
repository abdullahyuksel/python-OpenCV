# -*- coding: utf-8 -*-

import cv2
img = cv2.imread("images.jpeg")

gausian = cv2.GaussianBlur(img, (5,5), 0)

median = cv2.medianBlur(img, 5)

bilatera = cv2.bilateralFilter(img, 9, 75, 75)

blur = cv2.blur(img, (5,5))

cv2.imshow("img",img)
cv2.imshow("gaussian",gausian)
cv2.imshow("median",median)
cv2.imshow("bilatera",bilatera)
cv2.imshow("blur",blur)
cv2.waitKey()

cv2.destroyAllWindows()