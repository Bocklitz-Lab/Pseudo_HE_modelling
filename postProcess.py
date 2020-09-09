# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 12:22:38 2019

@author: si62qit
"""
from scipy import interpolate
import numpy as np
from skimage.morphology import disk
from skimage import filters
from skimage.transform import resize

#%%
"""
This section post-processes generated images
"""

def mean_filtering(img, disk_factor = 5):
    
    filtered_img = np.zeros(img.shape, dtype= np.uint8)
    
    if img.shape[2] > 1:
        filtered_img[:,:,0] = filters.rank.mean(img[:,:,0], disk(disk_factor))
        filtered_img[:,:,1] = filters.rank.mean(img[:,:,1], disk(disk_factor))
        filtered_img[:,:,2] = filters.rank.mean(img[:,:,2], disk(disk_factor))
    else:
        filtered_img = filters.rank.mean(img, disk(disk_factor))
        
    return filtered_img

def gaussian_filtering(img, sigma = 5):
    
    if img.shape[2] > 1:
        filtered_img = filters.gaussian(img, sigma = sigma, multichannel = True)
    else:
        filtered_img = filters.gaussian(img, sigma = sigma, multichannel = False)
    
    return filtered_img

def interpolate_image(img, patch_size = 256, radius = 2, method='nearest'):
    
    checker_pattern = np.ones((img.shape[0], img.shape[1]), dtype = np.float64)

    for rowstart in range(0, img.shape[0], patch_size): 
        for colstart in range(patch_size, img.shape[1], patch_size) :
            checker_pattern[rowstart:rowstart+patch_size, colstart-radius:colstart+radius] = np.nan
            
    for rowstart in range(patch_size, img.shape[0], patch_size): 
        for colstart in range(0, img.shape[1], patch_size) :
            checker_pattern[rowstart-radius:rowstart+radius, colstart:colstart+patch_size] = np.nan
    
    x = np.arange(0, checker_pattern.shape[1])
    y = np.arange(0, checker_pattern.shape[0])
    
    #mask invalid values
    checker_pattern = np.ma.masked_invalid(checker_pattern)
    xx, yy = np.meshgrid(x, y)
    
    #get only the valid values
    x1 = xx[~checker_pattern.mask]
    y1 = yy[~checker_pattern.mask]
    newarr_0 = img[:,:,0][~checker_pattern.mask]
    newarr_1 = img[:,:,1][~checker_pattern.mask]
    newarr_2 = img[:,:,2][~checker_pattern.mask]

    restored = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype = np.uint8)
    restored[:,:,0] = interpolate.griddata((x1, y1), newarr_0.ravel(), (xx, yy), method=method)
    restored[:,:,1] = interpolate.griddata((x1, y1), newarr_1.ravel(), (xx, yy), method=method)
    restored[:,:,2] = interpolate.griddata((x1, y1), newarr_2.ravel(), (xx, yy), method=method)
        
    return restored

def remove_checkerboard(img, patch_size = 256, radius = 2, method='nearest'):

    slice_size = 4*patch_size
    restored = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype= np.uint8)    
    
    for i in range(0, img.shape[0], slice_size):
        for j in range(0, img.shape[1], slice_size):
            restored[i:i+slice_size, j:j+slice_size, :] = interpolate_image(img[i:i+slice_size, j:j+slice_size, :], patch_size, radius = 3, method='linear')

    return restored

    