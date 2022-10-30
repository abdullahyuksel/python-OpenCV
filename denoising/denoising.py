import cv2
"""Parametreler:
P1 - Kaynak Görüntü Dizisi
P2 - Hedef Görüntü Dizisi
P3 - Ağırlıkları hesaplamak için kullanılan şablon yamasının piksel cinsinden boyutu.
P4 - Verilen piksel için ağırlıklı ortalamayı hesaplamak için kullanılan pencerenin piksel cinsinden boyutu.
P5 - Parlaklık bileşeni için filtre gücünü düzenleyen parametre.
P6 - Yukarıdakinin aynısı ancak renkli bileşenler için // Gri tonlamalı bir görüntüde kullanılmaz.
10,10,7,15 """

img = cv2.imread("bear.png")

denoising = cv2.fastNlMeansDenoisingColored(img,None,10,20,7,25)


cv2.imshow("denoising",denoising)
cv2.imshow("bear",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

