import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*3/5)
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])
 
"""
Finds polynomial fit for lines
"""
def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        
        ## Negative slope -> left lane line
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    ## Average the fits
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)

    ## Get line arrays
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)

    return np.array([left_line, right_line])

"""
Returns canny of image (gradient lines)
"""
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    ## Use Gaussian blur method
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    
    ## Make lines showing gradient
    canny = cv2.Canny(blur, 50, 150)
    return canny

"""
Creates image showing lines
"""
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
             cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

"""
Find gradient lines over a triangular region
"""
def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


########## Image ##########
# image = cv2.imread('test_image.jpg')
# 
# ## Make gray copy to not edit original
# lane_image = np.copy(image)
# canny_image = canny(lane_image)
# cropped_image = region_of_interest(canny_image)
# 
# ## Detect lines in canny image
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# averaged_lines = average_slope_intercept(lane_image, lines)
# 
# line_image = display_lines(lane_image, averaged_lines)
# 
# ## Put lines over lane image, weighted for clarity
# combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
# 
# cv2.imshow("result", combo_image)
# cv2.waitKey(0)

########## Video ##########
cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _, frame = cap.read()

    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    
    ## Detect lines in canny image
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    
    line_image = display_lines(frame, averaged_lines)
    
    ## Put lines over lane image, weighted for clarity
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    
    cv2.imshow("result", combo_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    
