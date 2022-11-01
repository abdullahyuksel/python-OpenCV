import cv2
import dlib


detector = dlib.get_frontal_face_detector()
model = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

def mid(p1,p2):
    return(  int(((p1[0]+p2[0])/2)),   int((p1[1]+p2[1])/2))

f = open("dataset.csv","a")

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
        print(mburun)

        cv2.circle(frame,(po_ust_sol[0],po_ust_sol[1]),3,(255,0,0),-1)
        cv2.circle(frame,(po_alt_sol[0],po_alt_sol[1]),3,(0,0,255),-1)
        

    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xff == ord("q"):
        r = input("işlem girin : ")
        if r == "q":
            break
        elif r == "v":
            sol = input("sol için veri girin : ")
            sag = input("sağ için veri girin : ")
            f.write(f"{sol_mesafe},{sag_mesafe},{mburun},{sol},{sag}\n")
        
cap.release()
cv2.destroyAllWindows()





