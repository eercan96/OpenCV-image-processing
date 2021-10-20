import cv2,os
import numpy as np
from PIL import Image

tanıyıcı = cv2.face.LBPHFaceRecognizer_create()
cascadeyolu = "face.xml"
faceCascade = cv2.CascadeClassifier(cascadeyolu)
path = 'yuzverileri'

def get_images_and_labels(yol):
     image_paths = [os.path.join(yol, f) for f in os.listdir(yol)]
     images = []
     labels = []
     for image_path in image_paths:
         image_pil = Image.open(image_path).convert('L')
         image = np.array(image_pil, 'uint8')
         nbr = int(os.path.split(image_path)[1].split(".")[0].replace("face-", ""))
         print(nbr)
         faces = faceCascade.detectMultiScale(image)
         for (x, y, w, h) in faces:
             images.append(image[y: y + h, x: x + w])
             labels.append(nbr)
             cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
             cv2.waitKey(10)
     return images, labels


images, labels = get_images_and_labels(path)
cv2.imshow('test',images[0])
cv2.waitKey(1)

tanıyıcı.train(images, np.array(labels))
tanıyıcı.write('training/trainer.yml')
cv2.destroyAllWindows()