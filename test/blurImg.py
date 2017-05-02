"""
script pour fabriquer des images avec flou de mouvement a partir d'un original
"""

import cv2
import numpy as np
img = cv2.imread('originale.png')
#  kernel = np.ones((5,5),np.float32)/25
ker1 = np.matrix('0. 0. 0.;1. 1. 1.;0. 0. 0.')/3
flou = cv2.filter2D(img,-1,ker1)
cv2.imwrite("flou1.png", flou)

ker2 = np.matrix('0. 1. 0.;0. 1. 0.;0. 1. 0.')/3
flou = cv2.filter2D(img,-1,ker2)
cv2.imwrite("flou2.png", flou)

ker3 = np.matrix('1. 0. 0.;0. 1. 0.;0. 0. 1.')/3
flou = cv2.filter2D(img,-1,ker3)
cv2.imwrite("flou3.png", flou)
