# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

import warnings
warnings.filterwarnings("ignore")

batchSize=50
#eğitime girecek resim partisi sayısı
epochSize=30
#resimler kaç kere eğitileceği

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

path = "specialData"
#hata datamızın yolunu belirtiyoruz

myList = os.listdir(path)
#hata datamızı listeye atıyoruz

numOfFails = len(myList)
#hata listemizin uzunluğunu alıyoruz

print("Hata tür sayısı: ",numOfFails)
#kac hata turumuz var alıyoruz

images = []
imageType = []
#resim ve türler için dizi oluşturuyoruz

for i in range(numOfFails): # klasör içinde dolaşıyoruz
    myImageList = os.listdir(path + "\\" + str(i)) # türlerin isimlerini aldık
    for j in myImageList: # türler içinde dolaşıyoruz
        img = cv2.imread(path + "\\" + str(i) + "\\" + j) # resiölerin isimlerini aldık
        img = cv2.resize(img, (32,32)) # resimleri 32,32 yaptık
        images.append(img) #images listesine ekledik
        imageType.append(i) # tür listesine ekledik


#görselleştirme        
# =============================================================================
# print(len(images))
# print(len(imageType))
# #sayıları gördük
# =============================================================================

images = np.array(images)
imageType = np.array(imageType)
# geri kalan işlemlerde array a çevirmemiz gerekiyor


#veriyi ayırma

x_train, x_test, y_train, y_test = train_test_split(images, 
                                                    imageType, 
                                                    test_size = 0.5, 
                                                    random_state = 42)
#train ile test verilerimizi yarı yarıya ayırdık. random state 42 parametresi deneme ile bulduk.
x_train, x_validation, y_train, y_validation = train_test_split(x_train, 
                                                                y_train, 
                                                                test_size = 0.2, 
                                                                random_state = 42)
#train ile test verilerimizi %20-%80 ayırdık. random state 42 parametresi deneme ile bulduk.
print(images.shape)
print(x_train.shape)
print(x_test.shape)
print(x_validation.shape)
# ayrılmış halleriyle sayıları ve boyutları görelim

#görselleştirme
# =============================================================================
# img = preProcess(x_train[1])
# img = cv2.resize(img, (300,300))
# cv2.imshow("preProcess", img)
# =============================================================================


#ön işlemden geçirme

x_train = np.array(list(map(preProcess, x_train)))
#map metodu 2 parametre alır 1. fonksiyondur bu fonksiyonu 2. parametre olan veri listesinin hepsine uygular
#sonra bu işlenmiş verileri liste yapıyoruz bu listeyi x_traim e atıyoruz
x_test = np.array(list(map(preProcess, x_test)))
#aynı işlemi x_test için yaptık  
x_validation = np.array(list(map(preProcess, x_validation)))
#aynı işlemi validation için yaptık


# ön işlem sonrası boyutlandırma

x_train = x_train.reshape(-1, 32, 32, 1)
#bu resimleri reshape yapıyoruz -1 in anlamı ne kadar resm varsa hepsine uygula demektir
x_test = x_test.reshape(-1, 32, 32 ,1)
#aynı işlemi teste uygula
x_validation = x_validation.reshape(-1, 32, 32 ,1)
#aynı işlemi validationa uygula

print(x_train.shape)
print(x_test.shape)
print(x_validation.shape)
#boyutları görelim(ilk olarak traine uygulamadım train harici tek boyutlu çıktı)


#data generate

dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.1,
                             rotation_range=10)
# genişleterek, uzatarak, yakınlaştırarak, çevirerek yeni veriler üretiyoruz.
dataGen.fit(x_train)
#bu data jenaratörünü x_train resimleri için çalıştırıyoruz


#kategorilere ayırma

y_train = to_categorical(y_train, numOfFails)
#train verilerini hata kategorilerine ayırıyoruz
y_test = to_categorical(y_test, numOfFails)
#test verilerini hata kategorilerine ayırıyoruz
y_validation = to_categorical(y_validation, numOfFails)
#validation verilerini hata kategorilerine ayırıyoruz


#eğitim modeli oluşturulması

model = Sequential()
#sequential bir temel oluşturuyoruz
model.add(Conv2D(input_shape = (32,32,1), # modele giren resimlerin boyutları
                 filters = 8, #8 adet filtre yapıyoruz
                 kernel_size = (5,5), #kernelimim 5 e 5 matris
                 activation = "relu", #aktivasyonu relu ile yapıyoruz
                 padding = "same"))#bir sıra piksel eklemeyi same padding ile yapıyoruz
#modele ekleme yapıyoruz

model.add(MaxPooling2D(pool_size=(2,2)))
#piksel ekleme yapıyoruz

model.add(Dropout(0.2))
#yukarıda yeni veri üretmiştik bu ezberlemeyi getirir(overfitting) bu nedenle syreltme yapıyoruz.

model.add(Conv2D(filters = 16, #16 adet filtre yapalım
                 kernel_size = (3,3), #kernelimiz 3 e 3 olsun
                 activation = "relu", #aktivasyonumuz relu
                 padding = "same"))# bir sıra piksel ekleyelim
#modele ikinci ekleme yaparken tekrar modele giren resimlerin boyutlarını söylemeye gerek yok

model.add(MaxPooling2D(pool_size=(2,2)))
#piksel ekleme yapıyoruz

model.add(Dropout(0.2))
#yukarıda yeni veri üretmiştik bu ezberlemeyi getirir(overfitting) bu nedenle syreltme yapıyoruz.

model.add(Conv2D(filters = 32, #16 adet filtre yapalım
                 kernel_size = (3,3), #kernelimiz 3 e 3 olsun
                 activation = "relu", #aktivasyonumuz relu
                 padding = "same"))# bir sıra piksel ekleyelim
#modele ikinci ekleme yaparken tekrar modele giren resimlerin boyutlarını söylemeye gerek yok

model.add(MaxPooling2D(pool_size=(2,2)))
#piksel ekleme yapıyoruz

model.add(Dropout(0.2))
#yukarıda yeni veri üretmiştik bu ezberlemeyi getirir(overfitting) bu nedenle syreltme yapıyoruz.
model.add(Flatten())
#düzleştirme yapıyoruz
model.add(Dense(units = 256, activation = "relu"))
#evrişim ağları ekliyoruz 256 hücre olsun aktivasyon olarak relu kullanalım
model.add(Dropout(0.2))
#tektar seyreltme yapalım
model.add(Dense(units = numOfFails, activation = "softmax"))
#çıktı katmanımızı yazıyoruz çıkışta hata türümüz kadar hücremiz olacak birine karar verecek 
#aktivasyon olarak softmax kullanıcaz

model.compile(loss = "categorical_crossentropy", #loss parametremiz kategoriselleştirme
              optimizer=("Adam"), #optimizer ımız adaptif momentum
              metrics=["accuracy"]) #değerlendirmemiz accuaracy
#modelimizi derliyoruz

#modelin eğitim aşaması

hist = model.fit_generator(dataGen.flow(x_train, y_train, batch_size = batchSize),
                           validation_data = (x_validation, y_validation),
                           epochs = epochSize, steps_per_epoch = x_train.shape[0]//batchSize, shuffle = 1)
#modelin çıktısının görselleştirilmesi için hist e atıyoruz
#resimler 15 kez eğitilecek her adımdaki eğitim resim sayısına kalansız bölünecek sekilde eğitilecek
#data jenaratör train üzerine uygulanacak

model.save("hataAlgilamaEgitim.h5")
#modeli kaydediyoruz


# degerlendirme

hist.history.keys()
# histogramın içindekileri görebiliriz. kayıplar ve doğrulukları yazacağız
plt.figure()
plt.plot(hist.history["loss"], label = "Eğitim Loss")
#loss da kaybı görücez
plt.plot(hist.history["val_loss"], label = "Val Loss")
#doğrulama kaybını görücez
plt.legend()
plt.show()


plt.figure()
plt.plot(hist.history["accuracy"], label = "Eğitim accuracy")
#doğruluğu görücez
plt.plot(hist.history["val_accuracy"], label = "Val accuracy")
#doğrulama doğrularını görücez
plt.legend()
plt.show()


#sonuc

score = model.evaluate(x_test, y_test, verbose = 1)
#0. adeste kayıp, 1. adreste kazanç yazacak bunu görselleştir diyoruz
print("Test loss: ", score[0])
#test kaybı yazıcaz
print("Test accuracy: ", score[1])
#test doğruluğu yazıcaz

y_pred = model.predict(x_validation)
#validation sonuçlarına bakıcaz
y_pred_class = np.argmax(y_pred, axis = 1)
#max değerine bakıcaz
y_true = np.argmax(y_validation, axis = 1)
#gercek degere bakıcaz
cm = confusion_matrix(y_true, y_pred_class)
#y true ile y predict i karşılaştırıcaz 

f, ax = plt.subplots(figsize = (8,8))
#8 e 8 piksel tablo oluşturucaz
sns.heatmap(cm, annot = True, linewidths=0.01, cmap="Greens", linecolor= "gray", fmt = ".1f", ax = ax)
#heat map oluşturduk.
plt.xlabel("predicted")
plt.ylabel("true")
plt.title("cm")
plt.show() 