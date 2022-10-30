import cv2

img = cv2.imread("image.jpeg",0)

_,thresh = cv2.threshold(img,75,255,cv2.THRESH_BINARY)

athresoshold_mean = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,3)
athresoshold_gaussian = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,3)

cv2.imshow("adaptiveThreshold gausian",athresoshold_gaussian)
cv2.imshow("adaptive thresh mean",athresoshold_mean)
cv2.imshow("thresh",thresh)
cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

