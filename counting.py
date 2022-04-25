import cv2
import numpy as np
import sys

img = cv2.imread(sys.argv[1])

cv2.imshow('Segmented Image', img)

white_pix = np.sum(img == 255)
black_pix = np.sum(img == 0)

print('Number of white pixels:', white_pix)
print('Number of black pixels:', black_pix)