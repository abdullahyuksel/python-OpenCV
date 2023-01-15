# -*- coding: utf-8 -*-

import cv2
from keras.models import Sequential, load_model
import numpy as np

#preprocess

def preProcess(img):
    #siyah beyaza ceviriyoruz
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #histogramı 0-255 yaptık
    img = cv2.equalizeHist(img)
    #görüntünün piksel değerlerini 0-1 arasına getirdik
    img = img / 255
    #son hali geri döndürdük
    return img

cap = cv2.VideoCapture(0)
#kamerayı açıyoruz
cap.set(3,480)
cap.set(4,480)
#genişlik ve yükseklik ayarlandı

model = load_model("hataAlgilamaEgitim.h5")
#modelimizi yüklüyoruz


while True:

    success, frame = cap.read()
    #kameradan frame alıyoruz

    if success: #basarılı ise
        img = np.asarray(frame)
        #frame i array a ceviriyoruz
        img = cv2.resize(img, (32,32))
        #eğitimimizi 32,32 yalmıştık buradada buna dikkat ediyoruz
        img = preProcess(img)
        #resimleri daha hızlı algılamak için ön işleme fonksiyonumuz ile 0-1 arasına getirdik
        img = img.reshape(1,32,32,1)
        # resmi istenen boyuta getirdik
        
        predictions = model.predict(img)
        # modele göre tahmin yapıyor
        
        failIndex = np.argmax(predictions)
        # bu tahminde gelen sayıların toplamı 1 oluyor. argmax en büyük olanın indeksini veriyor
        
        probVal = np.amax(predictions)
        #probilistik değerleri alıyoruz çünki görselleştirmede kullanacağız

        print(failIndex, probVal)
        #hata türü olasılığı yazacak str(failIndex) + 

        if probVal > 0.7:
            if failIndex == 0:
                cv2.putText(frame, "Dedektor Yok    " + str(probVal), (50,50), cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0), 1)
            if failIndex == 1:
                cv2.putText(frame, "Led Cubuk Hatasi    " + str(probVal), (50,50), cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0), 1)
            if failIndex == 2:
                cv2.putText(frame, "Leke Hatasi    " + str(probVal), (50,50), cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0), 1)
            #frame in üzerine olasılığı yazıcaz
        cv2.imshow("siniflandirma", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break
