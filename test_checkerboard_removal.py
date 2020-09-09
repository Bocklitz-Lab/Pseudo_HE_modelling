# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:43:41 2020

@author: si62qit
"""
from postProcess import remove_checkerboard
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np

path = 'C:/Users/si62qit/Documents/PhDJenaPranita/codesPython/WP6_HE_modelling/6_Pix2Pix_vs_cycleGAN/cycleGAN/results_v2/'
img = imread(path+'ID_003.png')

#%%
img_1 = img[768:1280, 768:1280, :]
plt.title("Original patch")
plt.imshow(img_1)

post_img_1, chk = remove_checkerboard(img_1[:,:,0:3], patch_size = 256, radius = 1, method='nearest', downscale=False)
plt.figure()
plt.title("nearest, Radius 1")
plt.imshow(post_img_1[:,:,0:3])

post_img_1, chk = remove_checkerboard(img_1[:,:,0:3], patch_size = 256, radius = 2, method='nearest', downscale=False)
plt.figure()
plt.title("nearest, Radius 2")
plt.imshow(post_img_1[:,:,0:3])

post_img_1, chk = remove_checkerboard(img_1[:,:,0:3], patch_size = 256, radius = 3, method='nearest', downscale=False)
plt.figure()
plt.title("nearest, Radius 3")
plt.imshow(post_img_1[:,:,0:3])

post_img_1 = remove_checkerboard(img_1[:,:,0:3], patch_size = 256, radius = 1, method='linear', downscale=False)
plt.figure()
plt.title("Linear, Radius 1")
plt.imshow(post_img_1[:,:,0:3])

post_img_1 = remove_checkerboard(img_1[:,:,0:3], patch_size = 256, radius = 2, method='linear', downscale=False)
plt.figure()
plt.title("Linear, Radius 2")
plt.imshow(post_img_1[:,:,0:3])

post_img_1 = remove_checkerboard(img_1[:,:,0:3], patch_size = 256, radius = 3, method='linear', downscale=False)
plt.figure()
plt.title("Linear, Radius 3")
plt.imshow(post_img_1[:,:,0:3])

#%%
patch_size = 256
slice_size = 4*patch_size
restored = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype= np.uint8)    
    
for i in range(0, img.shape[0], slice_size):
   for j in range(0, img.shape[1], slice_size):
       restored[i:i+slice_size, j:j+slice_size, :] = remove_checkerboard(img[i:i+slice_size, j:j+slice_size, :], patch_size, radius = 3, method='linear', downscale=False)

plt.imsave(path+'postProcess/ID_002.png', restored)     