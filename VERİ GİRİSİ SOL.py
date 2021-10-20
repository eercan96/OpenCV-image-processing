import cv2
cam = cv2.VideoCapture(0)
tarayıcı=cv2.CascadeClassifier('face.xml')
i=0
offset=50
kisi_id=input('KİMLİK NUMARASI GİRİNİZ...')
while True:
    _, img =cam.read()
    gri=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    yüzler=tarayıcı.detectMultiScale(gri, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in yüzler:
        i=i+1
        cv2.imwrite("yuzverileri/kisi-" + kisi_id + '.' + str(i) + ".jpg", gri[y:y + h , x :x + w])
        cv2.rectangle(img, (x-offset , y-offset), (x + w +offset, y + h+offset), (225, 0, 0), 2)
        cv2.imshow('resim', img[y :y + h, x :x + w])
        cv2.waitKey(100)
    if i>9:
        cam.release()
        cv2.destroyAllWindows()
        break
