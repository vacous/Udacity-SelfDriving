# -*- coding: utf-8 -*-
"""
Created on Fri May 19 16:32:17 2017

@author: Administrator
"""
# camera calibration 
import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import glob 

CAM_IMGS = glob.glob('./camera_cal/calibration*.jpg')
SAMPLE_IMG = mpimg.imread('./test_images/straight_lines1.jpg')
IMG_SHAPE = SAMPLE_IMG.T.shape[1:3]


CORNER_X = 9
CORNER_Y = 6
ACTUAL_POS = np.zeros((CORNER_X * CORNER_Y, 3), np.float32)
ACTUAL_POS[:,:2] = np.mgrid[0:CORNER_X, 0:CORNER_Y].T.reshape(-1,2) # fill positon idx 

obj_points = []
img_points = []

for each_img in CAM_IMGS:
    img = mpimg.imread(each_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    [ret, corners] = cv2.findChessboardCorners(gray, (CORNER_X,CORNER_Y), None)    
    if ret:
        img_points.append(corners)
        obj_points.append(ACTUAL_POS)


RET, MTX, DIST, RVECS, TVECS =  cv2.calibrateCamera(obj_points, img_points,
                                                        IMG_SHAPE, None, None)  
# calculate retval, cameraMatrix, distCoeffs, rvecs, tvecs      
BOTTOM_LEFT =  [256,682]
BOTTOM_RIGHT = [1040,682]
PIC_POS = np.float32([[804,527], BOTTOM_RIGHT, BOTTOM_LEFT, [479,527]])
DES_POS = np.float32([[BOTTOM_RIGHT[0], PIC_POS[0][1]], BOTTOM_RIGHT,
                      BOTTOM_LEFT, [BOTTOM_LEFT[0],PIC_POS[-1][1]]])
PERSP_M = cv2.getPerspectiveTransform(PIC_POS, DES_POS)
INVER_M = cv2.getPerspectiveTransform(DES_POS, PIC_POS)

def uncamAndUnwrap(read_in_img, in_MTX, in_DIST, in_PERSP_M):
    '''
    takes an img correct with camera matrix and perspective matrix 
    '''
    uncam = cv2.undistort(read_in_img, in_MTX, in_DIST, None, in_MTX) # undistort cam
    # apply s channel 
    hls_img = cv2.cvtColor(uncam, cv2.COLOR_RGB2HLS)
    s_img = hls_img[:,:,2]
    
    s_thresh_min = 100
    s_thresh_max = 255
    s_binary = np.zeros_like(s_img)
    s_binary[(s_img >= s_thresh_min) & (s_img <= s_thresh_max)] = 1


    # apply sobel operator 
    sobel_x = cv2.Sobel(s_binary, cv2.CV_64F, 1, 0)     
    abs_sobel = np.absolute(sobel_x)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    thresh_min = 10
    thresh_max = 70
    sx_binary = np.zeros_like(scaled_sobel)
    sx_binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    
    
    # combine two img 
    combined_binary = np.zeros_like(sx_binary)
    combined_binary[(s_binary == 1) | (sx_binary == 1)] = 1 

    
    img_shape = read_in_img.T.shape[1:3]
    wraped = cv2.warpPerspective(combined_binary, in_PERSP_M, img_shape, flags = cv2. INTER_LINEAR)
    
#    plt.figure()
#    plt.imshow(s_binary)
#    
#    
#    plt.figure()    
#    plt.title("test")
#    plt.imshow(sx_binary, cmap='gray')
#    
#    
#    
#    plt.figure()
#    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
#    ax1.set_title('Stacked thresholds')
#    ax1.imshow(color_binary)
#    
#    ax2.set_title('Combined S channel and gradient thresholds')
#    ax2.imshow(combined_binary, cmap='gray')
#    
#    
#    plt.figure()
#    plt.imshow(wraped)
    return wraped

# test on example image 
in_img = mpimg.imread('./test_images/test2.jpg')
test_img = uncamAndUnwrap(in_img, MTX, DIST, PERSP_M)
#test_gray = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)

#
#uncamAndUnwrap()
#
#
## find the gradient peaks 
#histogram = np.sum(combined_binary[combined_binary.shape[0]//2:,:], axis=0)
#plt.figure()
#plt.plot(histogram)




def windowFitInitial(in_grad_img, ym_per_pix, xm_per_pix):
    '''
    takes a gradient img 
    returns two fit lines of the curve coefficeints 
    this function is from udaccity 
    '''
    binary_warped = in_grad_img
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 50
    # Set minimum number of pixels found to recenter window
    minpix = 100
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # find the real world fit line and curvature 
    # convert to real world unit
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    y_eval = max(np.max(lefty), np.max(righty))
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    # use the max curvauture of the two and correct the fit function that has lower curvature
    if left_curverad * 0.8 < right_curverad:
        right_fit[2] = right_fit[0] * y_eval**2 + right_fit[1] * y_eval + right_fit[2] - (left_fit[0] * y_eval**2 + left_fit[1] * y_eval)
        right_fit[0] = left_fit[0]
        right_fit[1] = left_fit[1]
        out_curvature = left_curverad
    elif right_curverad * 0.8 < left_curverad:
        left_fit[2] = left_fit[0] * y_eval**2 + left_fit[1] * y_eval + left_fit[2] - (right_fit[0] * y_eval**2 + right_fit[1] * y_eval)
        left_fit[0] = right_fit[0]
        left_fit[1] = right_fit[1]
        out_curvature = right_curverad
    return left_fit, right_fit, out_curvature 


in_left, in_right, out_curve = windowFitInitial(test_img, 30/720, 3.7/700)


def combineLaneArea(ori_img, warped, left_fit, right_fit, max_y, Minv):
    '''
    taken from the udacity tips 
    '''
    ploty = np.array(range(max_y*100))*0.01
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (ori_img.shape[1], ori_img.shape[0])) 
    plt.imshow(newwarp)
    # Combine the result with the original image
    result = cv2.addWeighted(ori_img, 1, newwarp, 0.3, 0)
    plt.imshow(result)

combineLaneArea(in_img, test_img, in_left, in_right, IMG_SHAPE[1], INVER_M)













