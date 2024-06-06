# Proyecto de Detección de Rostros

Este proyecto utiliza OpenCV para detectar rostros y reconocer expresiones a partir de imágenes en tiempo real capturadas por la cámara.

## Captura y Almacenamiento de Rostros

El siguiente script captura imágenes de rostros detectados con la cámara y los almacena en un directorio específico para su posterior entrenamiento.

```python
import numpy as np
import cv2 as cv
import math

rostro = cv.CascadeClassifier('haarcascade_frontalface_alt2.xml')
cap = cv.VideoCapture(0)
i = 2039

while True:
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    rostros = rostro.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in rostros:
        frame2 = frame[y-10:y+h+10, x-10:x+w+10]
        cv.imshow('rostros2', frame2)
        frame2 = cv.resize(frame2, (100,100), interpolation=cv.INTER_AREA)
        cv.imwrite('C:\\Users\\axela\\caras\\Triste\\triste' + str(i) + '.png', frame2)
    cv.imshow('rostros', frame)
    i += 1
    k = cv.waitKey(1)
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()
```

## Entrenamiento del Reconocedor de Rostros

El código siguiente carga las imágenes de rostros almacenadas, las etiqueta y entrena un modelo para el reconocimiento.

```python
import cv2 as cv
import numpy as np
import os

dataSet = 'C:\\Users\\axela\\caras'
faces = os.listdir(dataSet)
labels = []
facesData = []
label = 0

for face in faces:
    facePath = dataSet + '\\' + face
    print(facePath)
    for faceName in os.listdir(facePath):
        labels.append(label)
        facesData.append(cv.imread(facePath + '\\' + faceName, 0))
    label += 1

faceRecognizer = cv.face.LBPHFaceRecognizer_create()
faceRecognizer.train(facesData, np.array(labels))
faceRecognizer.write('LBPH.xml')
```

## Detección y Reconocimiento de Rostros en Tiempo Real

Este script utiliza el modelo entrenado para reconocer rostros en tiempo real desde la cámara.

```python
import os
import cv2 as cv

faceRecognizer = cv.face.LBPHFaceRecognizer_create()
faceRecognizer.read('LBPH.xml')
cap = cv.VideoCapture(0)
rostro = cv.CascadeClassifier('haarcascade_frontalface_alt2.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cpGray = gray.copy()
    rostros = rostro.detectMultiScale(gray, 1.3, 3)
    for (x, y, w, h) in rostros:
        frame2 = cpGray[y:y+h, x:x+w]
        frame2 = cv.resize(frame2, (100,100), interpolation=cv.INTER_CUBIC)
        result = faceRecognizer.predict(frame2)
        if result[1] < 500:
            cv.putText(frame, '{}'.format(faces[result[0]]), (x, y-25), 2, 1.1, (0, 255, 0), 1, cv.LINE_AA)
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            cv.putText(frame, 'Desconocido', (x, y-20), 2, 0.8, (0, 0, 255), 1, cv.LINE_AA)
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv.imshow('frame', frame)
    k = cv.waitKey(5)
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()
```