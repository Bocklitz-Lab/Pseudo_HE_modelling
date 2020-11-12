# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:30:51 2019

@author: P.Pradhan

This script is used to train Pix2Pix model and use the trained model to predict
images from test dataset.
"""

# import all packages
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import tensorflow as tf
from keras import backend as K
K.set_image_dim_ordering('tf')
K.set_image_data_format("channels_last")
K.tensorflow_backend._get_available_gpus()

# config the script to run with GPU
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

from keras.models import load_model
import pandas as pd
from matplotlib import pyplot
from imageProcessing.patches import image_to_patch, patch_to_image, save_patches
from skimage.io import imread
import ntpath
import numpy as np
from utils.pix2pix import *
from postProcess import *
import time

# get working directory
path = "C:/Users/si62qit/Documents/PhDJenaPranita/pseudoHE"#os.getcwd()

# input pacth size
patch_size = 256

#%%  
# load image data
dataset = load_real_samples(path+'/3_all_results/3_GeneratedHE_pix2pix_loss_mae/mm_256.npz')#'/6_Pix2Pix_vs_cycleGAN/pix2pix/train.npz'
print('Loaded', dataset[0].shape, dataset[1].shape)
# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]
# define the models
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
# define the composite model
gan_model = define_gan(g_model, d_model, image_shape)
# summarize the model
gan_model.summary()
# train model
t = time.time()
train(d_model, g_model, gan_model, dataset)
print("--- %s seconds ---" % (time.time() - t))
    
#%%
# predict, reconstruct and plot HE stain image for all dataset
model = load_model('C:/Users/si62qit/Documents/PhDJenaPranita/pseudoHE/6_Pix2Pix_vs_cycleGAN/pix2pix/models/model_018300.h5') #model_047520

df = pd.read_csv(os.getcwd() + '/train.csv')
for i, item in df.iterrows():
    if i >= len(df):
        break
    src_img = imread('C:/Users/si62qit/Documents/PhDJenaPranita/pseudoHE/' + item[0])
    src_img = flip_contrast(src_img)
    src_img =  scale_sample(src_img)
    src_patch = image_to_patch(src_img[patch_size:src_img.shape[0]-patch_size, patch_size:src_img.shape[1]-patch_size], patch_size)  #image_to_patch(src_img, patch_size)  
    gen_patch = predict_patches(src_patch, model)
    save_patches(gen_patch, (patch_size, patch_size, 3), (ntpath.basename(item[0])).split('.png')[0], 'C:/Users/si62qit/Documents/PhDJenaPranita/pseudoHE/6_Pix2Pix_vs_cycleGAN/pix2pix/pred_HE_train/' )
    gen_image = patch_to_image(gen_patch, src_img.shape, 256)
    src_mask = imread(path + item[7])
    gen_image[src_mask==0] = 255
    pyplot.imsave(path+'/6_Pix2Pix_vs_cycleGAN/pix2pix/results/'+ntpath.basename(item[2]), gen_image)
