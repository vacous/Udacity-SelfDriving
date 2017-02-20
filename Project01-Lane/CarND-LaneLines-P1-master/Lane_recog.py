# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 22:34:03 2017

@author: Administrator
"""
import matplotlib.pyplot as plt
import numpy as np 
from lane_helper import lane_help

#reading in an image
lane_detect = lane_help()
lane_detect.grayscale('test_images\solidYellowCurve.jpg')
lane_detect.gaussian_blur(5)
lane_detect.canny(50,100)
INTEREST_REGION_TEST = [(100,539),(450, 320),(500, 320), (900,539)]
lane_detect.region_of_interest(INTEREST_REGION_TEST)
image = lane_detect.hough_lines(1,np.pi/180,20,10,1)
lane_detect.weighted_img()
