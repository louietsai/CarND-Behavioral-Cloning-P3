
#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)




import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

def preprocessing(img):
	gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
	# Define a kernel size and apply Gaussian smoothing
	kernel_size = 5
	blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
	# Define our parameters for Canny and apply
	low_threshold = 70
	high_threshold = 200
	edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
	# Next we'll create a masked edges image using cv2.fillPoly()
	mask = np.zeros_like(edges)   
	ignore_mask_color = 255   
	# This time we are defining a four sided polygon to mask
	imshape = image.shape
	vertices = np.array([[(imshape[1]*0/100,imshape[0]*4/5),(imshape[1]/2, imshape[0]*35/100), (imshape[1]/2, imshape[0]*35/100), (imshape[1],imshape[0]*60/100)]], dtype=np.int32)
	cv2.fillPoly(mask, vertices, ignore_mask_color)
	masked_edges = cv2.bitwise_and(edges, mask)
	# Define the Hough transform parameters
	# Make a blank the same size as our image to draw on
	rho = 2 # distance resolution in pixels of the Hough grid
	theta = np.pi/180 # angular resolution in radians of the Hough grid
	threshold = 30     # minimum number of votes (intersections in Hough grid cell)
	min_line_length = 100 #minimum number of pixels making up a line
	max_line_gap = 200    # maximum gap in pixels between connectable line segments
	line_image = np.copy(image)*0 # creating a blank to draw lines on
	# Run Hough on edge detected image
	# Output "lines" is an array containing endpoints of detected line segments
	lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),min_line_length, max_line_gap)

	# Iterate over the output "lines" and draw lines on a blank image
	for line in lines:
		for x1,y1,x2,y2 in line:
			cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

	# Create a "color" binary image to combine with line image
	color_edges = np.dstack((edges, edges, edges))
	
	# Draw the lines on the edge image
	lines_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0) 
	return lines_edges


if __name__ == "__main__":
    import os
    imgfiles = os.listdir("run1/")
    for img in imgfiles:
        #reading in an image
        imgpath = "run1/"+img
        # Read in and grayscale the image
        image = mpimg.imread(imgpath)
        lines_edges = preprocessing(image)
        plt.imshow(lines_edges)
        line_imgpath = "run1_out/"+"final_"+img
        plt.savefig(line_imgpath)




