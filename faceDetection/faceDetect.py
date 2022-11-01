# -*- coding: utf-8 -*-

import cv2
import dlib

detector = dlib.get_frontal_face_detector()

cap = cv2.VideoCapture(1)

while True:
    _, frame = cap.read()
    
    faces = detector(frame)
    #print(faces)
    
    for face in faces:
        x = face.left()
        y = face.top()
        w = face.right()
        h = face.bottom()
        cv2.rectangle(frame, (x,y), (w,h), (0,0,255), 2)
    
    cv2.imshow("frame",frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()