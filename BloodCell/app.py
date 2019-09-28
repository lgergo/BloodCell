import numpy as np
import cv2 as cv

img=cv.imread("C:/Users/Gergo/OneDrive/Pictures/Canyonland Utah.jpg")
cv.imshow("image",img)
cv.waitKey(0)
cv.destroyAllWindows()