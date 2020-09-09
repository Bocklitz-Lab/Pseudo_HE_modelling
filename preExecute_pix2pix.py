# -*- coding: utf-8 -*-
"""
Created on Wed May 02 11:32:06 2018

@author: si62qit
"""

"""
To create a .csv file of file names of images and labels. 

"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import ntpath
import pandas as pd
import os
from skimage.io import imread
from imageProcessing.patches import image_to_patch, save_patches
from utils.pix2pix import normalize_intensity, scale_patches, square_root_img, flip_contrast

#%%
# get path of the .csv file
path = os.getcwd()

# set the patch of save directory
save_folder = path + '/6_Pix2Pix_vs_cycleGAN/pix2pix/'

# set the patch size
patch_size = 256

def prep_data(mode):
    
    src_list = []
    tar_list = []
    image_dim = []
    image_ID = []
    
    df = pd.read_csv(mode + '.csv')
        
    for i, item in df.iterrows():
        if i >= len(df):
            break
        # read source and target images
        src_img, tar_img = imread(path + item[0]) , imread(path + item[6])
        
        # invert the contrast of source image
        src_img = flip_contrast(src_img)
        
        # remove the registration artefact on the edges from the target image
        xx =  tar_img.shape[0]-patch_size
        yy =  tar_img.shape[1]-patch_size
        
        # print the min and max intensity values from source and target image
        print(np.min(src_img), np.max(src_img))
        print(np.min(tar_img), np.max(tar_img))
        
        # generate patches of size 256*256 from source and target image
        src_patch = image_to_patch(src_img[patch_size:xx, patch_size:yy], patch_size)    
        tar_patch = image_to_patch(tar_img[patch_size:xx, patch_size:yy], patch_size)  
        
        # save the target and source patches
        save_patches(src_patch, (patch_size, patch_size, 3), (ntpath.basename(item[0])).split('.png')[0], save_folder+ 'MM_train/' )
        save_patches(tar_patch, (patch_size, patch_size, 3), (ntpath.basename(item[0])).split('.png')[0], save_folder+ 'HE_train/' )
        
        # append the list of patches
        src_list.append(src_patch)
        tar_list.append(tar_patch)
        
        # store metadata of source image
        image_dim.append(np.shape(src_img)[:3])
        image_ID.append(np.repeat(i, len(src_patch)))
        
    src_list, tar_list, image_dim, image_ID = np.concatenate(src_list), np.concatenate(tar_list), np.array(image_dim), np.concatenate(image_ID)
        
    return [src_list, tar_list, image_dim, image_ID]

# load dataset
[src_images, tar_images, dim, im_ID] = prep_data(path + '/train')
print('Loaded: ', src_images.shape, tar_images.shape)

# save as compressed numpy array
filename = path+ '/6_Pix2Pix_vs_cycleGAN/pix2pix/train'
np.savez_compressed(filename, src_images, tar_images)
print('Saved dataset: ', filename)  

# save as compressed numpy array
filename = path+ '/6_Pix2Pix_vs_cycleGAN/pix2pix/train_metadata'
np.savez_compressed(filename, dim, im_ID)
print('Saved dataset: ', filename) 


