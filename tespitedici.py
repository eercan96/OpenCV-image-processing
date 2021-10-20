import cv2
import time
import datetime

tanıyıcı=cv2.face.LBPHFaceRecognizer_create()
tanıyıcı.read('training/trainer.yml')
cascadeyolu = "face.xml"
faceCascade = cv2.CascadeClassifier(cascadeyolu)
yol = 'yuzverileri'
cam = cv2.VideoCapture(0)
while True:
    ret, im =cam.read()
    gri=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gri, scaleFactor=1.5, minNeighbors=8, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in faces:
        tahminEdilenKisi, conf = tanıyıcı.predict(gri[y:y + h, x:x + w])
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        if(tahminEdilenKisi == 2):
             tahminEdilenKisi= 'Anil Dursun'
        elif (tahminEdilenKisi == 1):
            tahminEdilenKisi = 'Emre Ercan'
        elif (tahminEdilenKisi == 3):
            tahminEdilenKisi = 'f'
        else:
            tahminEdilenKisi= "BILINMEYEN KISI"
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.75
        fontColor = (255, 255, 255)
        cv2.putText(im, str(tahminEdilenKisi), (x, y + h), fontFace, fontScale, fontColor)
        cv2.imshow('im',im)
        print("GİRİŞ YADA ÇIKIŞ ZAMANI =", datetime.datetime.now().strftime("%y-%m-%d-%H.%M"))

        cv2.waitKey(10)
