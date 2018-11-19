import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
Returns canny of image (gradient lines)
"""
def canny(image):
    gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
    
    ## Use Gaussian blur method
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    
    ## Make lines showing gradient
    canny = cv2.Canny(blur, 50, 150)
    return canny
    
def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    return mask

image = cv2.imread('test_image.jpg')

## Make gray copy to not edit original
lane_image = np.copy(image)
canny = canny(lane_image)

cv2.imshow("result", region_of_interest(canny))
cv2.waitKey(0)
