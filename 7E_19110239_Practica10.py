import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import cv2

img = cv.imread('Cartas.jpg')

roi = cv2.selectROI(img)
print(roi)

sel = img[int(roi[1]):int(roi[1]+roi[3]),
           int(roi[0]):int(roi[0]+roi[2])]

cv2.imshow('Imagen Selecionada',sel)

mask = np.zeros(sel.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

rect = (0,0,300,300)

cv.grabCut(sel,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

sel = sel*mask2[:,:,np.newaxis]

plt.imshow(sel)
plt.colorbar()
plt.show()

