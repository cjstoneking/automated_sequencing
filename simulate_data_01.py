#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 11:34:06 2018

@author: root
"""

import numpy as np
import scipy as scp
import skimage.transform

#generate a sequence of image frames of rolonies
#output is array of dimension [x, y, frames]
#some details:
#rolonies are isotropic gaussians, full-width-at-half-max given by rolony_fwhm_in_microns
#they can have jitter from frame-to-frame, which is uniformly distributed on [-jitter, +jitter] 
#they can be missing on any given frame with probability given by rolony_dropout
#(note that they can reappear on subsequent frames)
#all parameters are specified on a scale of microns
#output image is downsampled according to specified resolution (microns_per_pixel)
def generate_rolony_images(fov_in_microns = (1000, 1000), microns_per_pixel = 2, n_frames = 2,
                                       mean_rolonies_per_frame = 100, rolony_fwhm_in_microns = 4, rolony_jitter_in_microns = 10,
                                       rolony_dropout = 0.1, rolony_integrated_signal = 2):
    
    image = np.zeros([fov_in_microns[0], fov_in_microns[1], n_frames])
    
    #construct rolony intensity profile in 2D
    rolony_sigma_in_microns = 2*np.sqrt(2*np.log(2))*rolony_fwhm_in_microns
    gx, gy = np.meshgrid(np.arange(-np.floor(3*rolony_sigma_in_microns), np.floor(3*rolony_sigma_in_microns), 1), 
                         np.arange(-np.floor(3*rolony_sigma_in_microns), np.floor(3*rolony_sigma_in_microns), 1))
    
    rolony_intensity_profile = 1/(np.sqrt(2*np.pi)*rolony_sigma_in_microns)*np.exp(-0.5*(gx**2 + gy**2)/rolony_sigma_in_microns**2)
    
    #sample number of rolonies and their locations at random
    n_rolonies = int(np.ceil(mean_rolonies_per_frame/(1 - rolony_dropout)))
    margin = int(np.ceil(len(gx)/2) + rolony_jitter_in_microns)
    
    rolony_centers  = np.concatenate([np.random.uniform(margin, image.shape[0]-margin, n_rolonies), np.random.uniform(margin, image.shape[1]-margin, n_rolonies)])
    rolony_centers  = np.transpose(rolony_centers.reshape([2, -1]))
    
    #add rolonies to individual frames
    for f in range(n_frames):
        rolony_present = np.random.binomial(1, 1 - rolony_dropout, size=n_rolonies)
        positions = rolony_centers + np.random.uniform(-rolony_jitter_in_microns, rolony_jitter_in_microns, rolony_centers.shape)
        for n in range(n_rolonies):
            if(rolony_present[n] == 1):
                x0 = int(positions[n, 0] - np.floor(rolony_intensity_profile.shape[0]/2))
                y0 = int(positions[n, 1] - np.floor(rolony_intensity_profile.shape[1]/2))
                x1 = int(x0 + rolony_intensity_profile.shape[0])
                y1 = int(y0 + rolony_intensity_profile.shape[1])
                image[x0:x1, y0:y1, f] = image[x0:x1, y0:y1, f] + rolony_intensity_profile
                
    #downsample the image
    if(microns_per_pixel > 1):
        downsampled_frame_list = list()
        for f in range(n_frames):
            downsampled_frame_list.append(skimage.transform.downscale_local_mean(image[:,:, f], (microns_per_pixel, microns_per_pixel)))
        image_downsampled = np.zeros([downsampled_frame_list[0].shape[0], downsampled_frame_list[0].shape[1], n_frames])
        for f in range(n_frames):
            image_downsampled[:,:, f] = downsampled_frame_list[f]
        return image_downsampled
        
    else:
        return image