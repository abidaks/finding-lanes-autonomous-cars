import cv2

import numpy as np

import matplotlib.pyplot as plt

# This function is used to define edges in the image which are used to detect lanes
def define_edges(image):
	#converting the image from 3-dimension color image to 1-dimension image for furthur processing
	gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	#apply 5x5 gaussian blur filter to fiter the noise from image
	blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
	#finding the gradient of the image using canny filter for edge detection
	canny_image = cv2.Canny(blur_image, 50, 150)
	return canny_image

# This function identify the area of interest in an image where we need to focus.
def identify_area_of_interest(image):
	height = image.shape[0]
	#currently coordinates are defined manually which need to be detected automatically
	polygons = np.array([[(200, height), (1000, height), (600, 300)]])
	mask_image = np.zeros_like(image)
	cv2.fillPoly(mask_image, polygons, 255)
	masked_image = cv2.bitwise_and(mask_image, image)
	return masked_image

# This function converts slope and intercept to coordinates of the line.
def make_coordinates(image, line_parameters):
	slope, intercept = line_parameters
	y1 = image.shape[0]
	y2 = int(y1*(3/5))
	x1 = int((y1 - intercept)/slope)
	x2 = int((y2 - intercept)/slope)
	return np.array([x1, y1, x2, y2])

# There were multiple lines detecting on right side and left side of the car.
# This function combines multiple lines and show only one line on left and one on right side of the car.
def slope_intercept_avg(image, lines):
	left_lines = []
	right_lines = []
	for line in lines:
		x1, y1, x2, y2 = line.reshape(4)
		parameters = np.polyfit((x1,x2), (y1,y2), 1)
		slope = parameters[0]
		intercept = parameters[1]
		if slope < 0:
			left_lines.append((slope, intercept))
		else:
			right_lines.append((slope, intercept))
	left_line = []
	right_line = []
	if left_lines:
		left_lines_avg = np.average(left_lines, axis=0)
		left_line = make_coordinates(image, left_lines_avg)
	if right_lines:
		right_lines_avg = np.average(right_lines, axis=0)
		right_line = make_coordinates(image, right_lines_avg)

	return np.array([left_line, right_line])

# This function is drawing lines on a blank image drived from the given image
def display_lines(image, lines):
	lines_image = np.zeros_like(image)
	if lines is not None:
		for line in lines:
			if len(line) > 0:
				x1, y1, x2, y2  = line.reshape(4)
				cv2.line(lines_image, (x1, y1), (x2, y2), (255,0,0), 10)

	return lines_image


cap = cv2.VideoCapture("data/lane_video.mp4")

while(cap.isOpened()):
	_, normal_image = cap.read()
	edged_image = define_edges(normal_image)
	interested_area_image = identify_area_of_interest(edged_image)
	# We are detecting multiple lines in the region of interest using openCV HoughLinesP function.
	straight_lines = cv2.HoughLinesP(interested_area_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
	average_lines = slope_intercept_avg(normal_image, straight_lines)
	image_with_lines = display_lines(normal_image, average_lines)
	combined_image = cv2.addWeighted(normal_image, 0.8, image_with_lines, 1, 1)
	cv2.imshow("Lanes Identified Video", combined_image)
	if cv2.waitKey(1) == ord("b"):
		break
cap.release()
cv2.destroyAllWindows()
