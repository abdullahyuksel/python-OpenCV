# -*- coding: utf-8 -*-

import cv2

img1 = cv2.imread("bit1.png")
img2 = cv2.imread("bit2.png")

andOp = cv2.bitwise_and(img1, img2)
cv2.imshow("and",andOp)

orOp = cv2.bitwise_or(img1, img2)
cv2.imshow("or", orOp)

xorOp = cv2.bitwise_xor(img1, img2)
cv2.imshow("xor", xorOp)

notOp = cv2.bitwise_not(img1)
cv2.imshow("not_bit1", notOp)

notOp2 = cv2.bitwise_not(img2)
cv2.imshow("not_bit2", notOp2)

cv2.imshow("bit1", img1)
cv2.imshow("bit2", img2)

cv2.waitKey(0)
cv2.destroyAllWindows()
