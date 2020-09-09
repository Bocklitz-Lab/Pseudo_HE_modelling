# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:30:51 2019

@author: si62qit
"""
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

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

from keras.models import load_model
import pandas as pd
from matplotlib import pyplot
from imageProcessing.patches import image_to_patch, patch_to_image
from skimage.io import imread
import ntpath
import numpy as np
from utils.pix2pix import *
from postProcess import *
import time

path = os.getcwd()
patch_size = 256

#%%  
# load image data
dataset = load_real_samples(path+'/6_Pix2Pix_vs_cycleGAN/pix2pix/train.npz')
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
# predict, reconstruct and plot HE stain for all dataset
model = load_model(path+'/6_Pix2Pix_vs_cycleGAN/pix2pix/models/model_047520.h5')

df = pd.read_csv(path + '/data.csv')
for i, item in df.iterrows():
    if i >= len(df):
        break
    src_img = imread(path + item[0])
    src_img = flip_contrast(src_img)
    src_img =  scale_sample(src_img)
    src_patch = image_to_patch(src_img, patch_size)  
#    src_patch = scale_patches(src_patch)
    gen_patch = predict_patches(src_patch, model)
    gen_image = patch_to_image(gen_patch, src_img.shape, 256)
    src_mask = imread(path + item[7])
    gen_image[src_mask==0] = 255
    pyplot.imsave(path+'/6_Pix2Pix_vs_cycleGAN/pix2pix/results/'+ntpath.basename(item[2]), gen_image)



