import numpy as np
import cv2
import matplotlib.pyplot as plt

#input
im1 = cv2.imread('newborn.tif')
plt.subplot(1,2,1)
plt.imshow(im1)
im1=im1.astype(np.double)
#rgb2gray
im1=(im1[:,:,0]+im1[:,:,1]+im1[:,:,2])/3
im2=im1

#n grayscale
n=4
for k in range(n):
    #grayscale range
    level1=255/n*(k)
    level2=255/n*(k+1)
    
    for i in range(im1.shape[0]):
        for j in range(im1.shape[1]):
            if im1[i,j]<level2 and im1[i,j]>=level1:
                im2[i,j]=level1

plt.subplot(1,2,2)
plt.imshow(im2, cmap = 'gray')
im2=cv2.normalize(im2, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
#output
cv2.imwrite('newborn2.tif',im2)
