#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 15:11:08 2018

@author: root
"""

import util_01 as util
import numpy as np
import numpy.fft
import skimage.transform
import scipy.ndimage.interpolation

def get_default_params():
    params = {}
    params.downsampling_factor = 1
    params.angle_range = np.arange(0, 360, 10)
    #the angle range to search over

#register one 2D image to another using rigid transformations
#(rotation, translation)
def rigid_registration(img1, img2, params):
    
    
    #assume both images are square, and same size
    
    

    
    if(np.abs(params.downsampling_factor - 1) > 10**(-6)):
        
        ds = (params.downsampling_factor, params.downsampling_factor)
        
        img1 = skimage.transform.downscale_local_mean(img1, ds)
        img2 = skimage.transform.downscale_local_mean(img2, ds)
        
    s = int(np.floor(1.0/np.sqrt(2)*img1.shape[0]))
    #side length of a square that is inscribed in a circle that is inscribed in the image
    #this is the maximum size square for which we have complete image data under all rotations
    
    i0 = int((img1.shape[0] - s)/2)
    i1 = i0 + s
    
    img1_test_region = img1[i0:i1, i0:i1]
    
    img1_fft = numpy.fft.fft2(img1_test_region)
    
    min_error = np.inf
    best_shift = np.array([0, 0])
    best_angle = 0
        
    for angle in params.angle_range:
        rotated_img2 = scipy.ndimage.interpolation.rotate(img2, angle)
        #this is automatically padded so its dimension will be larger
        
        i0 = int((rotated_img2.shape[0] - s)/2)
        i1 = i0 + s 
        
        img2_test_region = rotated_img2[i0:i1, i0:i1]
        
        img2_fft = numpy.fft.fft2(img2_test_region)
        
        CC = np.fft.ifft2(img1_fft*np.conj(img2_fft))
        
        CC_flat = np.reshape()
        
        CC_peak_value  = np.amax(CC_flat)
        CC_peak_index  = np.argmax(CC_flat)
        
        CC_peak_loc= np.unravel_index(CC_peak_index, CC.shape)
        
        area = img1_fft.shape[0]*img1_fft.shape[1]
        rfzero = np.sum(np.power(np.abs(img1_fft), 2))/area
        rgzero = np.sum(np.power(np.abs(img2_fft), 2))/area

        error = np.abs(1.0 - CC_peak_value*np.conj(CC_peak_value)/(rgzero[0,0]*rfzero[0,0]));
        if(error < min_error):
            
            best_angle = angle
        
            if CC_peak_loc[0] > np.fix(img1_fft.shape[0]/2):
                best_shift[0] = CC_peak_loc[0] - img1_fft.shape[0] - 1;
            else:
                best_shift[0] = CC_peak_loc[0] - 1;

            if CC_peak_loc[1] > np.fix(img1_fft.shape[1]/2):
                best_shift[1] = CC_peak_loc[1]- img1_fft.shape[1] - 1;
            else:
                best_shift[1] = CC_peak_loc[1] - 1;
            
            
    #finished iteration over angles
    
    #transform img2 according to best angle and shift
    registered_img2 = util.rotate_and_shift(img2, best_angle, best_shift)
    
    return registered_img2
    

        
        
    
    
    
    
        