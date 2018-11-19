import cv2
import numpy as np

image = cv2.imread('test_image.jpg')

## Make gray copy to not edit original
lane_image = np.copy(image)
gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)

## Use Gaussian blur method
blur = cv2.GaussianBlur(gray, (5,5), 0)

## Make lines showing gradient
canny = cv2.Canny(blur, 50, 150)

cv2.imshow('result', canny)
cv2.waitKey(0)
