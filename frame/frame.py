import cv2
import numpy as np


img = np.zeros((600,800,3),np.uint8)

cv2.rectangle(img,(100,100),(500,500),(0,0,255),2)
cv2.line(img,(0,0),(800,600),(255,0,0),2)
cv2.circle(img,(400,300),100,(0,255,0),-1)
cv2.putText(img,"fps:60",(100,100),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)


cv2.imshow("frame",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
