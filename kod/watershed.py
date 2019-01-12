import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

from scipy import ndimage as ndi
from skimage.morphology import watershed, disk
from skimage.filters import rank

# from get_stats import get_confusion_matrix_stats

# 1) wczytanie pierwotnego obrazu
img = cv.imread('test1.png')
cv.imshow('oryginalny obraz', img)

# 2) zastosowanie CLAHE
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
l, a, b = cv.split(lab)
clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl = clahe.apply(l)
limg = cv.merge((cl,a,b))
final = cv.cvtColor(limg, cv.COLOR_LAB2BGR)

# 3) wyodrebnienie kanalu zielonego oraz konwersja do odcieni szarosci
g = final.copy()
g[:, :, 0] = 0
g[:, :, 2] = 0
gray = cv.cvtColor(g,cv.COLOR_BGR2GRAY)

# 4) lokalna segmentacja progowa
adaptive = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,13,4)

# 5) odszumianie filtrem medianowym
th_median = rank.median(adaptive, disk(2))

# 6) zastosowanie rozmycia gaussowskiego
blur = cv.GaussianBlur(th_median,(5,5),0)

# 7) operacje morfologiczne
kernel = np.array([[0,1,0],[1,1,1],[0,1,0]],np.uint8)
th_dilate = cv.dilate(cv.bitwise_not(blur), kernel, iterations = 1)
th_open = cv.morphologyEx(th_dilate,cv.MORPH_OPEN,kernel, iterations = 1)

# 8) kolejne odszumianie filtrem medianowym
denoised = rank.median(th_open, disk(3))

# 9) binaryzacja - przygotowanie obrazu do wyliczenia wartosci sensitivity,
# specifity, accuracy
ret,th2 = cv.threshold(denoised,127,255,cv.THRESH_BINARY)

# 10) utworzenie markera, gradientu
markers = rank.gradient(denoised, disk(2)) < 5
markers = ndi.label(markers)[0]
gradient = rank.gradient(denoised, disk(2))
# zastosowanie segmentacji wododzialowej
labels = watershed(gradient, markers)

#===========================================================================
# wyswietlenie wynikow

fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(8, 8), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(gray, cmap=plt.cm.gray, interpolation='nearest')
ax[0].set_title('gray CLAHE')

ax[1].imshow(adaptive, cmap=plt.cm.gray, interpolation='nearest')
ax[1].set_title('adaptive')

ax[2].imshow(th_median, cmap=plt.cm.gray, interpolation='nearest')
ax[2].set_title('th_median')

ax[3].imshow(th_dilate, cmap=plt.cm.gray, interpolation='nearest')
ax[3].set_title('th_dilate')

ax[4].imshow(th_open, cmap=plt.cm.gray, interpolation='nearest')
ax[4].set_title('th_open')

ax[5].imshow(denoised, cmap=plt.cm.gray, interpolation='nearest')
ax[5].set_title('denoised')

ax[6].imshow(gradient, cmap=plt.cm.gray, interpolation='nearest')
ax[6].set_title('gradient')

ax[7].imshow(markers, cmap=plt.cm.nipy_spectral, interpolation='nearest')
ax[7].set_title('markers')

ax[8].imshow(labels, cmap=plt.cm.nipy_spectral, interpolation='nearest')
ax[8].set_title('labels')

for a in ax:
    a.axis('off')

fig.tight_layout()
plt.show()
