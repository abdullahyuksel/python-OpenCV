# -*- coding: utf-8 -*-
import cv2
import face_recognition
import dlib

detector = dlib.get_frontal_face_detector()


image_A = face_recognition.load_image_file("abdullah.PNG")
image_A_encoding = face_recognition.face_encodings(image_A)[0]

image_B = face_recognition.load_image_file("asaf.jpeg")
image_B_encoding = face_recognition.face_encodings(image_B)[0]

cap = cv2.VideoCapture(0)



while True:
    _,frame = cap.read()
    face_locations = []

    faces = detector(frame)
    for face in faces:
        x = face.left()
        y = face.top()
        w = face.right()
        h = face.bottom()
        face_locations.append((y,w,h,x))

    #faces_locations = face_recognition.face_locations(frame)
    
    faces_encodigs = face_recognition.face_encodings(frame,face_locations)

    i = 0

    for face in faces_encodigs:
        y,w,h,x = face_locations[i]
        i+=1
        resultA = face_recognition.compare_faces([image_A_encoding],face)
        resultB = face_recognition.compare_faces([image_B_encoding],face)
        if resultA[0] == True:
            cv2.rectangle(frame,(x,y),(w,h),(0,0,255),2)
            cv2.rectangle(frame,(x,h),(w,h+30),(0,0,255),-1)
            cv2.putText(frame,"ABDULLAH",(x,h+25),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
        elif resultB[0] == True:
            cv2.rectangle(frame,(x,y),(w,h),(0,0,255),2)
            cv2.rectangle(frame,(x,h),(w,h+30),(0,0,255),-1)
            cv2.putText(frame,"ASAF",(x,h+25),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
        else:
            cv2.rectangle(frame,(x,y),(w,h),(0,0,255),2)
            cv2.rectangle(frame,(x,h),(w,h+30),(0,0,255),-1)
            cv2.putText(frame,"TANIMSIZ",(x,h+25),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)




    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xff == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
