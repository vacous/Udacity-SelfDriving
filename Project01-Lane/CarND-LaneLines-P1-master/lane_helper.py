# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 13:35:56 2017

@author: Administrator
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

class lane_help():
    def __init__(self):
        pass
        
    def grayscale(self,img_address):
        '''
        apply grey tansform
        '''
        self.image = mpimg.imread(img_address)
        self.after_gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        print self.image.size
        return self.after_gray
    
    def gaussian_blur(self, kernel_size):
        '''
        Applies a Gaussian Noise kernel
        '''
        self.after_blur = cv2.GaussianBlur(self.after_gray, (kernel_size, kernel_size), 0)
        return self.after_blur    
    
    def canny(self, low_threshold, high_threshold):
        '''
        Applies the Canny transform
        '''
        self.edges = cv2.Canny(self.after_blur, low_threshold, high_threshold)
        return self.edges
    
    def region_of_interest(self, vertices):
        '''
        apply image mask 
        vertices are arrays of tuples, the array can be unsorted 
        '''
        # sort the array by its x coordinates 
        sorted(vertices, key = lambda each: each[0])
        mask = np.zeros_like(self.edges)
        ignore_mask_color = 255
        vertices = np.array([vertices])
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        self.masked_edges = cv2.bitwise_and(self.edges, mask)
        return self.masked_edges
      
    def hough_lines(self, rho, theta, threshold, min_line_len, max_line_gap):
        """
        `img` should be the output of a Canny transform.
            
        Returns an image with hough lines drawn.
        """
        img = self.masked_edges
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
        line_img = np.copy(self.image)*0
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(line_img, (x1,y1), (x2,y2),(255,0,0),5)
        self.line_img = line_img
    
    def weighted_img(self, alpha=0.8, beta=1., omega=0):
        """
        `img` is the output of the hough_lines(), An image with lines drawn on it.
        Should be a blank image (all black) with lines drawn on it.
        
        `initial_img` should be the image before any processing.
        
        The result image is computed as follows:
        
        initial_img * α + img * β + λ
        NOTE: initial_img and img must be the same shape!
        """
        lines_edges = cv2.addWeighted(self.line_img, alpha, self.image, beta, 0) 
        plt.imshow(lines_edges)
        
        