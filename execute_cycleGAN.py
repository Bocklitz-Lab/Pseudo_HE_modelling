# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 15:14:00 2019

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
from imageProcessing.patches import image_to_patch, patch_to_image, save_patches
from skimage.io import imread
import ntpath
import numpy as np
from utils.cycleGAN import *
from postProcess import *
import time

path = os.getcwd()

# input shape
image_shape = (256,256,3)

#%%
# train cyclegan models
def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA,
dataset):
    # define properties of the training run
    n_epochs, n_batch, = 100, 1
    
    # determine the output square shape of the discriminator
    n_patch = d_model_A.output_shape[1]
    
    # unpack dataset
    trainA, trainB = dataset
    
    # prepare image pool for fakes
    poolA, poolB = list(), list()
    
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    
    j = 0
    # manually enumerate epochs
    for i in range(n_steps):
        
        # change i value
        if (j >= len(trainA)):
            j = 0
            
        # select a batch of real samples
        X_realA, y_realA = generate_real_samples(j, trainA, n_batch, n_patch, mode = 'random') #j,mode = 'None'
        X_realB, y_realB = generate_real_samples(j, trainB, n_batch, n_patch, mode = 'random') #j,mode = 'None'
        
        # generate a batch of fake samples
        X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, n_patch)
        X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, n_patch)
        
        # update fakes from pool
        X_fakeA = update_image_pool(poolA, X_fakeA)
        X_fakeB = update_image_pool(poolB, X_fakeB)
        
        # update generator B->A via adversarial and cycle loss
        g_loss2, _, _, _, _ = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA,
        X_realA, X_realB, X_realA])
    
        # update discriminator for A -> [real/fake]
        dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
        dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)
        
        # update generator A->B via adversarial and cycle loss
        g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB,
        X_realB, X_realA, X_realB])
    
        # update discriminator for B -> [real/fake]
        dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
        dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)
        
        # summarize performance
        print('>%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (i+1, dA_loss1,dA_loss2,
        dB_loss1,dB_loss2, g_loss1,g_loss2))
        
        # evaluate the model performance every so often
        if (i+1) % (bat_per_epo * 1) == 0:
            # plot A->B translation
            summarize_performance(i, g_model_AtoB, trainA, 'AtoB')
            # plot B->A translation
            summarize_performance(i, g_model_BtoA, trainB, 'BtoA')
        if (i+1) % (bat_per_epo * 5) == 0:
            # save the models
            save_models(i, g_model_AtoB, g_model_BtoA)
            
        j = j+1
                
#%%
# load a dataset as a list of two arrays
dataset = load_real_samples(path+'/6_Pix2Pix_vs_cycleGAN/cycleGAN/train.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)
# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]
# generator: A -> B
g_model_AtoB = define_generator(image_shape)
# generator: B -> A
g_model_BtoA = define_generator(image_shape)
# discriminator: A -> [real/fake]
d_model_A = define_discriminator(image_shape)
# discriminator: B -> [real/fake]
d_model_B = define_discriminator(image_shape)
# composite: A -> B -> [real/fake, A]
c_model_AtoB = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
# composite: B -> A -> [real/fake, B]
c_model_BtoA = define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)
# train models
t = time.time()
train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset)
print("--- %s seconds ---" % (time.time() - t))

#%% predict, reconstruct and plot HE stain for all dataset
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

# load the models
cust = {'InstanceNormalization': InstanceNormalization}
model_AtoB = load_model('C:/Users/si62qit/Documents/PhDJenaPranita/pseudoHE/6_Pix2Pix_vs_cycleGAN/cycleGAN/models/g_model_AtoB_119640.h5', cust)
model_BtoA = load_model('C:/Users/si62qit/Documents/PhDJenaPranita/pseudoHE/6_Pix2Pix_vs_cycleGAN/cycleGAN/models/g_model_BtoA_134595.h5', cust)

df = pd.read_csv(path + '/data.csv')
for i, item in df.iterrows():
    if i >= len(df):
        break
    src_img_AtoB = imread(path + item[0])
    src_img_AtoB = flip_contrast(src_img_AtoB)
    src_img_AtoB =  scale_sample(src_img_AtoB)
    src_mask_AtoB = imread(path + item[7])
    src_patch_AtoB = image_to_patch(src_img_AtoB, image_shape[0])
    gen_patch_AtoB = predict_patches(src_patch_AtoB, model_AtoB)
#    save_patches(gen_patch_AtoB, image_shape, (ntpath.basename(item[0])).split('.png')[0], 'C:/Users/si62qit/Documents/PhDJenaPranita/pseudoHE/6_Pix2Pix_vs_cycleGAN/cycleGAN/pred_HE_train/' )
    gen_image_AtoB = patch_to_image(gen_patch_AtoB, src_img_AtoB.shape, image_shape[0])
    gen_image_AtoB[src_mask_AtoB==0] = 255
    pyplot.imsave(path+'/6_Pix2Pix_vs_cycleGAN/cycleGAN/results/'+ntpath.basename(item[2]), gen_image_AtoB)
    
    src_img_BtoA = imread(path + item[6])
    src_img_BtoA =  scale_sample(src_img_BtoA)
    src_patch_BtoA = image_to_patch(src_img_BtoA, image_shape[0])
    gen_patch_BtoA = predict_patches(src_patch_BtoA, model_BtoA)
    gen_image_BtoA = patch_to_image(gen_patch_BtoA, src_img_BtoA.shape, image_shape[0])
    gen_image_BtoA = flip_contrast(gen_image_BtoA)
    gen_image_BtoA[src_mask_AtoB==0] = 0
    pyplot.imsave(path+'/6_Pix2Pix_vs_cycleGAN/cycleGAN/reconstructed/'+ntpath.basename(item[2]), gen_image_BtoA)

#%% contrast adjust
from PIL import Image, ImageEnhance
df = pd.read_csv(path + '/data.csv')

for i, item in df.iterrows():
    if i >= len(df):
        break
    im = Image.open(path+'/6_Pix2Pix_vs_cycleGAN/cycleGAN/results/postProcess/02_remove_checkerboard/'+ntpath.basename(item[2]))
    mask = imread(path + item[7])
    enhancer = ImageEnhance.Contrast(im)
    im_output = enhancer.enhance(0.7) #decrease constrast
    im_output = np.array(im_output)
    im_output[mask==0] = 255
    pyplot.imsave(path+'/6_Pix2Pix_vs_cycleGAN/cycleGAN/results/postProcess/01_reduce_contrast/'+ntpath.basename(item[2]), im_output)

#%% remove checkerboard effect for generated images
    
df = pd.read_csv(path + '/data.csv')
for i, item in df.iterrows():
    if i >= len(df):
        break
    gen_image = imread(path+'/6_Pix2Pix_vs_cycleGAN/cycleGAN/results/'+ntpath.basename(item[2]))
    gen_image = remove_checkerboard(gen_image[:,:,0:3], patch_size = 256, radius = 3, method = 'linear')
    pyplot.imsave(path+'/6_Pix2Pix_vs_cycleGAN/cycleGAN/results/postProcess/02_remove_checkerboard/'+ntpath.basename(item[2]), gen_image)
    
#%%
# predict H&E patches and save
patch_size= 256
df = pd.read_csv(path + '/train.csv')
for i, item in df.iterrows():
    if i >= len(df):
        break
    src_img = imread('C:/Users/si62qit/Documents/PhDJenaPranita/pseudoHE/' + item[0])
    src_img = flip_contrast(src_img)
    src_img =  scale_sample(src_img)
    src_patch = image_to_patch(src_img[patch_size:src_img.shape[0]-patch_size, patch_size:src_img.shape[1]-patch_size], patch_size)  #image_to_patch(src_img, patch_size)  
    gen_patch = predict_patches(src_patch, model_AtoB)
    save_patches(gen_patch, (patch_size, patch_size, 3), (ntpath.basename(item[0])).split('.png')[0], 'C:/Users/si62qit/Documents/PhDJenaPranita/pseudoHE/6_Pix2Pix_vs_cycleGAN/cycleGAN/pred_HE_train/' )

