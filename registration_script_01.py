#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 09:47:52 2018

@author: cstoneki
"""

import simulate_data_01 as sim
import elastix_wrapper_01 as elastix
import util_01 as util
import rigid_registration_01 as rigidreg

import numpy as np

import matplotlib.pyplot as plt
import os

#settings
working_directory = '/Users/cstoneki/Documents/analysis/AutomatedSequencing'



                
def main(debug = False):
    image = sim.generate_rolony_images()
    frame = image[:,:,0]
    
    frame_dereg = util.rotate_and_shift(frame, 20, np.array([10, 15]))
    
    #try to register the two frames to each other
    
    frame_reg = rigidreg.rigid_registration(frame, frame_dereg, rigidreg.get_default_params())
    
    plt.imshow(frame)
    plt.show()
    plt.imshow(frame_dereg)
    plt.show()
    plt.imshow(frame_reg)
    plt.show()
    
    #params = elastix.get_default_params()
    #params.MaximumNumberOfIterations = 200
    #params.FinalGridSpacingInVoxels = 10
    
    ## Apply the registration (im1 and im2 can be 2D or 3D)
    #frame1_deformed = elastix.register(image[:,:,0], image[:,:,1], params, working_directory)
    
    #plt.imshow(frame1_deformed)
    #plt.show()
    


if __name__ == "__main__":

    main()
            