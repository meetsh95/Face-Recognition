# Face-Recognition

**Detection and dataset cretion**

```python

import cv2
import os
from cv2 import imread
from cv2 import imshow

#cam = cv2.VideoCapture(0)
#cam.set(3, 640)  # set video width
#cam.set(4, 480)  # set video height
img = cv2.imread('train/putin.jpg')
Cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # ret, img = cam.read()
    # img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = Cascade.detectMultiScale(gray, 1.5, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)


    imshow('detector', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:  # press 'ESC' to quit
        break
cv2.destroyAllWindows()

face_id = input('\n Enter the user id and press enter =  ')   # For each person, enter one numeric face id
print("\n Initializing face capture...")
count = 0

while (True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = Cascade.detectMultiScale(gray, 1.5, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])
        cv2.imshow('detector', img)

    k = cv2.waitKey(100) & 0xff

    if k == 27:
        break
    elif count >= 50:
        break

#cam.release()
print("\n Face captured.")

cv2.destroyAllWindows()

```

**Trainer**

```python
import cv2
import numpy as np
from PIL import Image
import os

path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
Cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def getImagesAndLabels(path):

    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = Cascade.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)

    return faceSamples, ids


print("\n Training faces. Wait for a few seconds..")

faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))
recognizer.write('trainer/trainer.yml')
print("\n {0} faces trained.".format(len(np.unique(ids))))
```


**Face Recognizer**
```python
import cv2
from cv2 import imread
from cv2 import imshow

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
Cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

font = cv2.FONT_HERSHEY_SIMPLEX
id = 0
names = ['None',
         'Shinzo Abe',
         'Vladimir Putin',
         'Sir Ajit Doval',
         'Tim Cook',
          ] # names related to ids: example ==> Abe: id=1,  etc

imgdet = imread('train/ajit.png')
imgrec = imread('test/ajit.jpg')


while True:

    graydet = cv2.cvtColor(imgdet, cv2.COLOR_BGR2GRAY)
    facesdet = Cascade.detectMultiScale(graydet, scaleFactor=1.7, minNeighbors=5, )
    for (x, y, w, h) in facesdet:
        cv2.rectangle(imgdet, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(imgdet, 'Trained image', (x + 120, y - 40), font, 1, (0, 255, 255), 2)
    imshow('detector', imgdet)

    grayrec = cv2.cvtColor(imgrec, cv2.COLOR_BGR2GRAY)
    facesrec = Cascade.detectMultiScale(grayrec, scaleFactor=1.4, minNeighbors=5, )   #minSize=(int(minW), int(minH)),)

    for (x, y, w, h) in facesrec:

        cv2.rectangle(imgrec, (x, y), (x + w, y + h), (0, 255, 0), 2)

        id, confidence = recognizer.predict(grayrec[y:y + h, x:x + w])

        # To Check if confidence is less than 100 ==> "0" is perfect match
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
            print("\n Confidence is ", confidence)
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(imgrec, 'Test image', (x + 230, y - 30), font,1,(0, 255, 255),2 )
        cv2.putText(imgrec, str(id), (x + 1, y - 5), font, 0.8, (255, 255, 255), 2)
        cv2.putText(imgrec, str(confidence), (x - 5, y + h + 30), font, 1, (255, 0, 0), 2)

    imshow('recognizer', imgrec)

    k = cv2.waitKey(30) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break

print("\n Exiting Program and cleanup stuff")
cv2.destroyAllWindows()
```
