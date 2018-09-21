#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Rigid registration implementation
Arguments:
    reference_img, deformed_img : templates (images that we base registration on)
    reference_img, img2: images that we transform according to registration
    these can be different, e.g. if we want to blur out points before reg,
    but don't want to blur final image
"""

import util_01 as util
import numpy as np
import numpy.fft
import skimage.transform
import scipy.ndimage.interpolation

def get_default_params():
    params = {}
    params['downsampling_factor'] = 1
    #whether to downsample prior to registration
    #usually not a good idea unless images are huge
    params['angle_range'] = np.arange(-180, 180, 10)
    #the angle range to search over

    return params

#register one 2D image to another using rigid transformations
#(for our purposes, these are transformations that can be expressed as rotation and translation)
#arguments: reference_img, deformed_img are the images used as templates
#we want to find the transformation that returns 
def rigid_registration(reference_img, deformed_img,  params=[]):
    
    
    #assume both images are square, and same size

    if(len(params)==0):
        params = get_default_params()

    
    if(np.abs(params['downsampling_factor'] - 1) > 10**(-6)):
        
        ds = (params['downsampling_factor'], params['downsampling_factor'])
        
        reference_img = skimage.transform.downscale_local_mean(reference_img, ds)
        deformed_img = skimage.transform.downscale_local_mean(deformed_img, ds)
        
        
    s = int(np.floor(1.0/np.sqrt(2)*reference_img.shape[0]))
    #side length of a square that is inscribed in a circle that is inscribed in the image
    #this is the maximum size square for which we have complete image data under all rotations
    
    i0 = int((reference_img.shape[0] - s)/2)
    i1 = i0 + s
    
    reference_img_test_region = reference_img[i0:i1, i0:i1]
    
    reference_img_fft = numpy.fft.fft2(reference_img_test_region)
    
    min_error = np.inf
    best_shift = np.array([0, 0])
    best_angle = 0
        
    for angle in params['angle_range']:
        rotated_deformed_img = scipy.ndimage.interpolation.rotate(deformed_img, angle)
        #this is automatically padded so its dimension will be larger
        
        i0 = int((rotated_deformed_img.shape[0] - s)/2)
        i1 = i0 + s 
        
        deformed_img_test_region = rotated_deformed_img[i0:i1, i0:i1]
        
        deformed_img_fft = numpy.fft.fft2(deformed_img_test_region)
        
        CC = np.fft.ifft2(reference_img_fft*np.conj(deformed_img_fft))
        
        CC_flat = np.reshape(CC, -1)
        
        CC_peak_value  = np.amax(CC_flat)

        
        area = reference_img_fft.shape[0]*reference_img_fft.shape[1]
        rfzero = np.sum(np.power(np.abs(reference_img_fft), 2))/area
        rgzero = np.sum(np.power(np.abs(deformed_img_fft), 2))/area

        error = np.abs(1.0 - CC_peak_value*np.conj(CC_peak_value)/(rgzero*rfzero));
        if(error < min_error):
            
            min_error = error
            
            best_angle = angle
            
            #use location of crosscorrelation peak to find shift
            CC_peak_index  = np.argmax(CC_flat)
            CC_peak_loc= np.unravel_index(CC_peak_index, CC.shape)
        
            if CC_peak_loc[0] > np.fix(reference_img_fft.shape[0]/2):
                best_shift[1] = CC_peak_loc[0] - reference_img_fft.shape[0] - 1;
            else:
                best_shift[1] = CC_peak_loc[0] - 1;

            if CC_peak_loc[1] > np.fix(reference_img_fft.shape[1]/2):
                best_shift[0] = CC_peak_loc[1]- reference_img_fft.shape[1] - 1;
            else:
                best_shift[0] = CC_peak_loc[1] - 1;
            
            
    #finished iteration over angles

    transform = {'angle':best_angle, 'shift':best_shift}
    #registered_img2 = util.rotate_and_shift(img2, best_angle, best_shift)
    
    return transform
    

        
        
    
    
    
    
        