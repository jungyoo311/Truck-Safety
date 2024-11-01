import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import numpy as np
import os
import cv2 
from moviepy.editor import VideoFileClip

file = '/night_laneChange_sample_crop.mp4'

def grayscale(img):
    """
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
    `vertices` should be a numpy array of integer points.
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


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
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
            
def line_intersection(line1, line2):
    """Find the intersection point of two lines, if they intersect."""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None  # Parallel lines

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

    if min(x1, x2) <= px <= max(x1, x2) and min(x3, x4) <= px <= max(x3, x4):
        return int(px), int(py)
    return None

def slope_lines(image, lines):
    if lines is None:
        # If no lines are detected, return the original image
        return image

    img = image.copy()
    poly_vertices = []
    order = [0, 1, 3, 2]

    left_lines = []  # Lines like /
    right_lines = []  # Lines like \

    all_lines = []  # Store all lines to check for intersections

    # Separate the lines into left and right based on their slope
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue  # Skip vertical lines

            m = (y2 - y1) / (x2 - x1)
            c = y1 - m * x1

            if m < 0:
                left_lines.append((m, c))
            elif m >= 0:
                right_lines.append((m, c))

    # Ensure left and right lines are not empty
    if len(left_lines) > 0:
        left_line = np.mean(left_lines, axis=0)
    else:
        left_line = None  # No left line detected

    if len(right_lines) > 0:
        right_line = np.mean(right_lines, axis=0)
    else:
        right_line = None  # No right line detected

    # Process the left and right lines if they exist
    for line in [left_line, right_line]:
        if line is None:
            continue  # Skip if no line was detected

        slope, intercept = line
        rows, cols = image.shape[:2]
        y1 = int(rows)
        y2 = int(rows * 0.6)

        # Equation of the line: x = (y - c) / m
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)

        new_line = [x1, y1, x2, y2]

        # Check for intersections with previously drawn lines
        for drawn_line in all_lines:
            intersection = line_intersection(new_line, drawn_line)
            if intersection:
                x2, y2 = intersection
                break  # Stop the line at the intersection point

        # Store the current line and draw it
        all_lines.append([x1, y1, x2, y2])
        draw_lines(img, np.array([[[x1, y1, x2, y2]]]))

        poly_vertices.append((x1, y1))
        poly_vertices.append((x2, y2))

    if len(poly_vertices) >= 4:
        poly_vertices = [poly_vertices[i] for i in order]
        cv2.fillPoly(img, pts=np.array([poly_vertices], 'int32'), color=(0, 255, 0))

    return cv2.addWeighted(image, 0.7, img, 0.4, 0.)



def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    #draw_lines(line_img, lines)
    line_img = slope_lines(line_img,lines)
    return line_img


def weighted_img(img, initial_img, α=0.1, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    lines_edges = cv2.addWeighted(initial_img, α, img, β, γ)
    #lines_edges = cv2.polylines(lines_edges,get_vertices(img), True, (0,0,255), 10)
    return lines_edges

def get_vertices(image):
    ysize, xsize = image.shape[:2]
    print(ysize, xsize)
    bottom_left  = [xsize*0.2, ysize*0.99]
    bottom_right = [xsize*0.8, ysize*0.99]
    apex    = [xsize*0.5, ysize*0.8]
    
    ver = np.array([[bottom_left, apex, bottom_right]], dtype=np.int32)
    return ver

def lane_finding_pipeline(image):
    
    #Grayscale
    gray_img = grayscale(image)
    #Gaussian Smoothing
    smoothed_img = gaussian_blur(img = gray_img, kernel_size = 5)
    #Canny Edge Detection
    canny_img = canny(img = smoothed_img, low_threshold = 150, high_threshold = 250)
    #Masked Image Within a Polygon
    masked_img = region_of_interest(img = canny_img, vertices = get_vertices(image))
    #Hough Transform Lines
    houghed_lines = hough_lines(img = masked_img, rho = 1, theta = np.pi/180, threshold = 20, min_line_len = 20, max_line_gap = 180)
    #Draw lines on edges
    output = weighted_img(img = houghed_lines, initial_img = image, α=0.8, β=1., γ=0.)
    
    return output


if __name__ == '__main__':
    white_output = './data/lane output/' + file[:-4] + '_output.mp4'
    clip1 = VideoFileClip(os.path.join("./data", "night_laneChange_sample_crop.mp4"))
    white_clip = clip1.fl_image(lane_finding_pipeline) 
    white_clip.write_videofile(white_output, audio=False)

    """
    For displaying canny edge threshold
    """
    # image = mpimg.imread('./data/test.png')
    # gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    # kernel_size = 5 
    # blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
    # blur_gray = (blur_gray*255).astype(np.uint8)
    # low_threshold = 180
    # high_threshold = 240
    # edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    # plt.imshow(edges, cmap='Greys_r')
    # plt.title("Canny Edge Detection Image")
    # plt.show()

    """
    For displaying the marking region in the picture
    """

    # image = mpimg.imread('./data/test.png')

    # ysize = image.shape[0]
    # xsize = image.shape[1]
    # print(ysize, xsize)

    # left_bottom = [xsize*0.2, ysize*0.99]
    # right_bottom = [xsize*0.8, ysize*0.99]
    # apex = [xsize*0.5, ysize*0.6]
    # fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
    # fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
    # fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

    # XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
    # region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
    #                     (YY > (XX*fit_right[0] + fit_right[1])) & \
    #                     (YY < (XX*fit_bottom[0] + fit_bottom[1]))
    
    # plt.imshow(image)
    # x = [left_bottom[0], right_bottom[0], apex[0], left_bottom[0]]
    # y = [left_bottom[1], right_bottom[1], apex[1], left_bottom[1]]
    # plt.plot(x, y, 'r--', lw=4)
    # plt.title("Region Of Interest")
    # plt.show()