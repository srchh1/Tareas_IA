import os
import cv2 as cv
import numpy as np

def redimensionar_imagen(img, i):
    """
    Redimensiona la imagen a 60x60 píxeles y la guarda.

    Args:
    - img: Imagen de entrada.
    - i: Índice para nombrar el archivo de imagen de salida.

    Returns:
    - None
    """
    frame2 = cv.resize(img, (60,60), interpolation = cv.INTER_AREA)
    cv.imwrite(f'C:/Users/axela/Desktop/IA/FindWally/sinwaldo/waldos{i}.png', frame2)

# Redimensionar imágenes en el directorio 'sinwaldo60'
i = 0
imgPaths = 'C:/Users/axela/Desktop/IA/FindWally/sinwaldo60'
nomFiles = os.listdir(imgPaths)
for nomFile in nomFiles:
    i += 1
    imgPath = f'{imgPaths}/{nomFile}'
    img = cv.imread(imgPath)
    redimensionar_imagen(img, i) 

def rotar_imagen(img, i):
    """
    Rota la imagen 360 grados y la guarda.

    Args:
    - img: Imagen de entrada.
    - i: Índice para nombrar el archivo de imagen de salida.

    Returns:
    - None
    """
    h, w = img.shape[:2]
    mw = cv.getRotationMatrix2D((h//2, w//2), 360, -1)
    img2 = cv.warpAffine(img, mw, (h,w))
    cv.imwrite(f'C:/Users/axela/Desktop/IA/FindWally/sinwaldo/waldos{i}.png', img2)

# Rotar imágenes en el directorio 'sinwaldo60'
i = 1230
imgPaths = 'C:/Users/axela/Desktop/IA/FindWally/sinwaldo60'
nomFiles = os.listdir(imgPaths)
for nomFile in nomFiles:
    i += 1
    imgPath = f'{imgPaths}/{nomFile}'
    img = cv.imread(imgPath)
    rotar_imagen(img, i) 

def espejar_imagen(img, i):
    """
    Voltea horizontalmente la imagen y la guarda.

    Args:
    - img: Imagen de entrada.
    - i: Índice para nombrar el archivo de imagen de salida.

    Returns:
    - None
    """
    img_espejo = cv.flip(img, 1)
    cv.imwrite(f'C:/Users/axela/Desktop/IA/FindWally/sinwaldo/waldos{i}.png', img_espejo)

# Crear imágenes espejo en el directorio 'sinwaldo60'
i = 0
imgPaths = 'C:/Users/axela/Desktop/IA/FindWally/sinwaldo60'
nomFiles = os.listdir(imgPaths)
for nomFile in nomFiles:
    i += 1
    imgPath = f'{imgPaths}/{nomFile}'
    img = cv.imread(imgPath)
    espejar_imagen(img, i) 

def convertir_a_gris(img, i):
    """
    Convierte la imagen a escala de grises y la guarda.

    Args:
    - img: Imagen de entrada.
    - i: Índice para nombrar el archivo de imagen de salida.

    Returns:
    - None
    """
    frame2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imwrite(f'C:/Users/axela/Desktop/IA/FindWally/n/wallygris{i}.png', frame2)

# Convertir imágenes a escala de grises en el directorio 'waldo'
i = 0
imgPaths = 'C:/Users/axela/Desktop/IA/FindWally/waldo'
nomFiles = os.listdir(imgPaths)
for nomFile in nomFiles:
    i += 1
    imgPath = f'{imgPaths}/{nomFile}'
    img = cv.imread(imgPath)
    convertir_a_gris(img, i) 

# Importar bibliotecas necesarias e inicializar el clasificador de cascada para Wally
import numpy as np
import cv2 as cv    
import math

wally = cv.CascadeClassifier('C:/Users/axela/Desktop/IA/FindWally/dataset/cascade.xml')

# Leer el archivo de imagen
frame = cv.imread('C:/Users/axela/Desktop/waldo3.jpg')

# Convertir la imagen a escala de grises
gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

# Detectar Wally
wallys = wally.detectMultiScale(gray, 1.1, 20)

# Dibujar rectángulos alrededor de los Wally detectados
for (x, y, w, h) in wallys:
    frame = cv.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)

# Guardar el resultado y mostrarlo
cv.imwrite('pruebas/imgs/wally7.jpg', frame)
cv.imshow('Wally', frame)

cv.waitKey(0)
cv.destroyAllWindows()