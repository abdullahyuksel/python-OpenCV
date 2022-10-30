import cv2
import numpy as np

img = cv2.imread("cicek.jpg")
#resim cok buyuk oldugu için yarıya indirdik
img = cv2.pyrDown(img)
#resmin hsv uzayını alıyoruz.
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#resimde filtreleme yapacağımız renk için sınırları belirliyoruz
low = np.array([23,100,50])
up = np.array([35,255,255])
#aldığımız hsv uzayında istediğimiz renk için oluşturdugumuz sınırlar ile maske olusturuyoruz
mask = cv2.inRange(hsv,low,up)
#opening ile dışarıdaki fazlalıkları, closing ile içerdeki fazlalıkları siliyoruz
kernel = np.ones((5,5),np.uint8)
mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
#median blur ile karışıklıkları maskeliyoruz
mask = cv2.medianBlur(mask,5)
#and ile orjinal resim ile maskeyi çarpıyoruz
fil = cv2.bitwise_and(img,img,mask=mask)

cv2.namedWindow("filter",cv2.WINDOW_NORMAL)
cv2.namedWindow("mask",cv2.WINDOW_NORMAL)
cv2.namedWindow("image",cv2.WINDOW_NORMAL)
cv2.imshow("filter",fil)
cv2.imshow("mask",mask)
cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
