import cv2
import dlib
import pandas as pd
from sklearn.linear_model import LinearRegression
import time


dataset = pd.read_csv("dataset.csv")


x = dataset.iloc[:,:3].values
y = dataset.iloc[:,3:].values


lr = LinearRegression()
lr.fit(x,y)


detector = dlib.get_frontal_face_detector()
model = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(1)

def mid(p1,p2):
    return(  int(((p1[0]+p2[0])/2)),   int((p1[1]+p2[1])/2))


while True:
    _,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = detector(frame)

    for face in faces:
        points = model(gray,face)
        points_list = [(p.x,p.y) for p in points.parts()]

        # sağ göz için
        p1_ust_sag,p2_ust_sag = points_list[37],points_list[38]
        p1_alt_sag,p2_alt_sag = points_list[41],points_list[40]

        po_ust_sag = mid(p1_ust_sag,p2_ust_sag)
        po_alt_sag = mid(p1_alt_sag,p2_alt_sag)

        sag_mesafe = po_alt_sag[1] - po_ust_sag[1]

        #sol göz için
        p1_ust_sol,p2_ust_sol = points_list[43],points_list[44]
        p1_alt_sol,p2_alt_sol = points_list[47],points_list[46]
        
        po_ust_sol = mid(p1_ust_sol,p2_ust_sol)
        po_alt_sol = mid(p1_alt_sol,p2_alt_sol)

        sol_mesafe = po_alt_sol[1]-po_ust_sol[1]


        #burun için
        mburun = points_list[30][1]-points_list[27][1]
        

        pred =  lr.predict([[sol_mesafe,sag_mesafe,mburun]])
        pred_list = []
        for i in pred:
            b = 0
            ik = 0
            if i[0] > 0.5:
                b =1
            else:
                b=0
            if i[1] > 0.5:
                ik=1
            else:
                ik=0
            pred_list.append([b,ik])
        print(pred_list)
        if pred_list == []:
            print("boş")
        else:
            print("boş değil")
            cv2.putText(frame,str(pred_list[0][0]),po_alt_sol,cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),1)
            cv2.putText(frame,str(pred_list[0][1]),po_alt_sag,cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),1)


    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xff == ord("q"):
        break
    

cap.release()
cv2.destroyAllWindows()





