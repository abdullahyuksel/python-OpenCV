import cv2
import numpy as np

img = cv2.imread("image.png")

kernel = np.ones((5,5),np.uint8)

erode = cv2.erode(img,kernel)

dilate = cv2.dilate(img,kernel)

opening = cv2.imread("opening.png")

opennmorp = cv2.morphologyEx(opening,cv2.MORPH_OPEN,kernel)



closeimage = cv2.imread("closing.png")

closingmorph = cv2.morphologyEx(closeimage,cv2.MORPH_CLOSE,kernel)


cv2.imshow("close image",closeimage)
cv2.imshow("closing morph",closingmorph)
cv2.imshow("morph open",opennmorp)
#cv2.imshow("dilate",dilate)
#cv2.imshow("erode",erode)
cv2.imshow("open",opening)
cv2.waitKey(0)
cv2.destroyAllWindows()



