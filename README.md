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
