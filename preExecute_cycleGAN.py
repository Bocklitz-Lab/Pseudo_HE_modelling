# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 16:59:33 2020

@author: si62qit
"""

import warnings
warnings.filterwarnings("ignore")
import ntpath
import pandas as pd
import os
import numpy as np
from skimage.io import imread
from imageProcessing.patches import image_to_patch, save_patches, filter_patches_homogeneity, filter_patches
from utils.cycleGAN import normalize_intensity, scale_patches, square_root_img, flip_contrast
import matplotlib.pyplot as plt

#%%

"""
Make patches data for cycleGAN, use high resolution HE images

"""
# get path of the .csv file
path = os.getcwd()

# set the patch of save directory
save_folder = path + '/6_Pix2Pix_vs_cycleGAN/cycleGAN/'

patch_size = 256

def prep_data(mode):
        
    df = pd.read_csv(mode + '.csv')
        
    for i, item in df.iterrows():
        if i >= len(df):
            break
        # read source and target images
        src_img, tar_img = imread(path + item[0]) , imread(path + item[4])[:,:,:3]
        
        # invert the contrast of source image
        src_img = flip_contrast(src_img)
        
#       generate patches of size 256*256 from source and target image
        src_patch = image_to_patch(src_img, patch_size)
        tar_patch = image_to_patch(tar_img, patch_size)
        
#       filter patches with homogeneity factor
        src_patch = filter_patches_homogeneity(src_patch, homogeneity=0.60)
        tar_patch = filter_patches_homogeneity(tar_patch, homogeneity=0.60)
        
        save_patches(src_patch, (patch_size, patch_size, 3), (ntpath.basename(item[0])).split('.png')[0], save_folder+ 'MM_train/' )
        save_patches(tar_patch, (patch_size, patch_size, 3), (ntpath.basename(item[0])).split('.png')[0], save_folder+ 'HE_train/' )

    return

prep_data(path + '/train') 

#%%
"""
Prepare data for cycleGAN

"""
# example of preparing the horses and zebra dataset
from os import listdir
from numpy import asarray
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed

# load all images in a directory into memory
def load_images(path, size=(256,256)):
    data_list = list()
    # enumerate filenames in directory, assume all are images
    for filename in listdir(path):
        # load and resize the image
        pixels = load_img(path + filename, target_size=size)
        # convert to numpy array
        pixels = img_to_array(pixels)
        # store
        data_list.append(pixels)
        
    return asarray(data_list)

# dataset path
path = os.getcwd() + '/3_GeneratedHE_cycleGAN_inverted_MM/'

# load dataset A
dataA1 = load_images(path + 'MM_train/')
dataA2 = load_images(path + 'MM_test/')
dataA = vstack((dataA1, dataA2))
print('Loaded dataA: ', dataA.shape)
# load dataset B
dataB1 = load_images(path + 'HE_train/')
dataB2 = load_images(path + 'HE_test/')
dataB = vstack((dataB1, dataB2))
print('Loaded dataB: ', dataB.shape)
# save as compressed numpy array
filename = os.getcwd() + '/6_Pix2Pix_vs_cycleGAN/cycleGAN/train.npz'
savez_compressed(filename, dataA, dataB)
print('Saved dataset: ', filename)