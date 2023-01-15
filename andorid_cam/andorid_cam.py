# -*- coding: utf-8 -*-

import cv2
import numpy as np
import requests

url = "http://192.168.1.37:8080/shot.jpg"

while True:
    img_resp = requests.get(url)
    
    img_arr = np.array(bytearray(img_resp.content), dtype = np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (480,480))
    
    cv2.imshow("android camera", img)
    
    
    if cv2.waitKey(1) & 0xff == ord("q"):
        break

cv2.destroyAllWindows()
    