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

batch_sizer=250

path = "myData"
#hata datamızın yolunu belirtiyoruz

myList = os.listdir(path)
#hata datamızı listeye atıyoruz

numOfClasses = len(myList)
#hata listemizin uzunluğunu alıyoruz

print("Label(sınıf) sayısı: ",numOfClasses)
#kac hata turumuz var alıyoruz

images = []
classNo = []
#resim ve türler için dizi oluşturuyoruz

for i in range(numOfClasses): # klasör içinde dolaşıyoruz
    myImageList = os.listdir(path + "\\" + str(i)) # türlerin isimlerini aldık
    for j in myImageList: # türler içinde dolaşıyoruz
        img = cv2.imread(path + "\\" + str(i) + "\\" + j) # resiölerin isimlerini aldık
        img = cv2.resize(img, (32,32)) # resimleri 32,32 yaptık
        images.append(img) #images listesine ekledik
        classNo.append(i) # tür listesine ekledik
        
print(len(images))
print(len(classNo))
#sayıları gördük

images = np.array(images)
classNo = np.array(classNo)
# geri kalan işlemlerde array a çevirmemiz gerekiyor

print(images.shape)
print(classNo.shape)
# sayıları ve boyutları görelim

#veriyi ayırma

x_train, x_test, y_train, y_test = train_test_split(images, 
                                                    classNo, 
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
  
    
x_train = np.array(list(map(preProcess, x_train)))
#map metodu 2 parametre alır 1. fonksiyondur bu fonksiyonu 2. parametre olan veri listesinin hepsine uygular
#sonra bu işlenmiş verileri liste yapıyoruz bu listeyi x_traim e atıyoruz
x_test = np.array(list(map(preProcess, x_test)))
#aynı işlemi x_test için yaptık  
x_validation = np.array(list(map(preProcess, x_validation)))
#aynı işlemi validation için yaptık

x_train = x_train.reshape(-1, 32, 32, 1)
#bu resimleri reshape yapıyoruz -1 in anlamı ne kadar resm varsa hepsine uygula demektir
x_test = x_test.reshape(-1, 32, 32 ,1)
#aynı işlemi teste uygula
x_validation = x_validation.reshape(-1, 32, 32 ,1)
#aynı işlemi validationa uygula
print(x_train.shape)
print(x_test.shape)
print(x_validation.shape)
#boyutları görelim


#data generate

dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.1,
                             rotation_range=10)
# genişleterek, uzatarak, yakınlaştırarak, çevirerek yeni veriler üretiyoruz.
dataGen.fit(x_train)
#bu data jenaratörünü x_train resimleri için çalıştırıyoruz

#kategorilere ayırma
y_train = to_categorical(y_train, numOfClasses)
#train verilerini kategorilere ayırıyoruz
y_test = to_categorical(y_test, numOfClasses)
#test verilerini kategorilere ayırıyoruz
y_validation = to_categorical(y_validation, numOfClasses)
#validation verilerini kategorilere ayırıyoruz

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

model.add(Conv2D(filters = 16, #16 adet filtre yapalım
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
#tektat seyreltme yapalım
model.add(Dense(units = numOfClasses, activation = "softmax"))
#çıktı katmanımızı yazıyoruz çıkışta hata türümüz kadar hücremiz olacak birine karar verecek 
#aktivasyon olarak softmax kullanıcaz

model.compile(loss = "categorical_crossentropy", #loss parametremiz kategoriselleştirme
              optimizer=("Adam"), #optimizer ımız adaptif momentum
              metrics=["accuracy"]) #değerlendirmemiz accuaracy
#modelimizi derliyoruz


#modelin eğitim aşaması

hist = model.fit_generator(dataGen.flow(x_train, y_train, batch_size = batch_sizer),
                           validation_data = (x_validation, y_validation),
                           epochs = 15, steps_per_epoch = x_train.shape[0]//batch_sizer, shuffle = 1)
#modelin çıktısının görselleştirilmesi için hist e atıyoruz
#resimler 15 kez eğitilecek her adımdaki eğitim resim sayısına kalansız bölünecek sekilde eğitilecek
#data jenaratör train üzerine uygulanacak

model.save("mnistWeights.h5")
#modeli kaydediyoruz

# degerlendirme
hist.history.keys()

plt.figure()
plt.plot(hist.history["loss"], label = "Eğitim Loss")
plt.plot(hist.history["val_loss"], label = "Val Loss")
plt.legend()
plt.show()


plt.figure()
plt.plot(hist.history["accuracy"], label = "Eğitim accuracy")
plt.plot(hist.history["val_accuracy"], label = "Val accuracy")
plt.legend()
plt.show()

#sonuc
score = model.evaluate(x_test, y_test, verbose = 1)
print("Test loss: ", score[0])
print("Test accuracy: ", score[1])

y_pred = model.predict(x_validation)
y_pred_class = np.argmax(y_pred, axis = 1)
y_true = np.argmax(y_validation, axis = 1)
cm = confusion_matrix(y_true, y_pred_class) 

f, ax = plt.subplots(figsize = (8,8))
sns.heatmap(cm, annot = True, linewidths=0.01, cmap="Greens", linecolor= "gray", fmt = ".1f", ax = ax)
plt.xlabel("predicted")
plt.ylabel("true")
plt.title("cm")
plt.show() 

    
    
    
    
    
    
    
    
    
    
    
    