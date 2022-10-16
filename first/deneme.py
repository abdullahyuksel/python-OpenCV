# -*- coding: utf-8 -*-

import cv2

resim = cv2.imread("kizkulesi.jpg",0)

cv2.imshow("resim", resim)

cv2.waitKey(0)

cv2.destroyAllWindows()