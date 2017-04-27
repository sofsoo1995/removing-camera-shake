"""
script pour appliquer des homographies a des images
"""

import numpy as np
import cv2

ima = cv2.imread("input_0.png")
print(ima.shape)
rows, cols = ima.shape[:2]
angle = np.array([15, 90])
for a in angle:
    M = cv2.getRotationMatrix2D((cols/2, rows/2), a, 1)
    print(M)
    alpha = np.arctan(cols/rows)
    R = np.sqrt((cols/2)**2 + (rows/2)**2)

    rotated90 = cv2.warpAffine(ima, M, (cols, rows))
    cv2.imwrite("rotated%d.png" % a, rotated90)
    
M = np.float32([[1, 0, 15], [0, 1, 5]])
translated15 = cv2.warpAffine(ima, M, (cols, rows))
cv2.imwrite("translated15.png", translated15)
