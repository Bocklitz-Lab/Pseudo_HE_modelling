# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 16:57:27 2020

@author: si62qit
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import glob
import matplotlib.pyplot as plt
from skimage.io import imread

#%%
    
"""
Run this section to crop images
"""
imageD = 'C:/Users/si62qit/Documents/PhDJenaPranita/codesPython/HEstaining_GAN/1_HE/*.png'

def crop_image(img,tol=0):
    # img is image data
    # tol  is tolerance
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]

for filename in glob.glob(imageD): #assuming gif
    img = imread(filename)
    plt.figure()
    plt.imshow(img[400:img.shape[0]-400, 400:img.shape[1]-400])
