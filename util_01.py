#!/usr/bin/env python2
# -*- coding: utf-8 -*-

#contains miscellaneous utility functions

import numpy as np

import scipy.ndimage.interpolation


def rotate_and_shift(img, angle, shift):
    
    rotated_img = scipy.ndimage.interpolation.rotate(img, angle)

    shifted_img = np.zeros(img.shape)
    
    orig_i0 = int(np.floor((rotated_img.shape[0] - img.shape[0])/2))
    orig_i1 = orig_i0 + img.shape[0]
    orig_j0 = int(np.floor((rotated_img.shape[1] - img.shape[1])/2))
    orig_j1 = orig_j0 + img.shape[1]
    
    trans_i0 = 0
    trans_i1 = shifted_img.shape[0]
    trans_j0 = 0
    trans_j1 = shifted_img.shape[1]
    
    orig_i0 = orig_i0 + shift[0]
    orig_i1 = orig_i0 + shift[0]
    orig_j0 = orig_j0 + shift[1]
    orig_j0 = orig_j0 + shift[1]
    
    if(orig_i0 < 0):
        d = 0 - orig_i0
        orig_i0 = 0
        trans_i0  = trans_i0 + d
    if(orig_j0 < 0):
        d = 0 - orig_j0
        orig_j0 = 0
        trans_j0  = trans_j0 + d
    
    if(orig_i1 > img.shape[0]):
        d = img.shape[0] - orig_i1
        orig_i1 = img.shape[0]
        trans_i1  = trans_i1 + d
    if(orig_j1 > img.shape[1]):
        d = img.shape[1] - orig_j1
        orig_j1 = img.shape[1]
        trans_j1  = trans_j1 + d
        
    shifted_img[trans_i0:trans_i1, trans_j0:trans_j1] = rotated_img[orig_i0:orig_i1, orig_j0:orig_j1]
    
    return shifted_img
    