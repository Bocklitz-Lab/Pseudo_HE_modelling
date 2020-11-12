# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 15:12:15 2019

@author: P.Pradhan

This script has cycleCGAN model.
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

# configure with GPU
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

# example of defining composite models for training cyclegan generators
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Activation
from keras.layers import LeakyReLU
from keras.initializers import RandomNormal
from keras.layers import Concatenate
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

from numpy.random import randint
import numpy as np
import matplotlib.pyplot as plt
from random import random
import math  

# specify image size
img_rows = img_cols = 256

#%%
# define the discriminator model
def define_discriminator(image_shape):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# source image input
	in_image = Input(shape=image_shape)
	# C64
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
	d = LeakyReLU(alpha=0.2)(d)
	# C128
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C256
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C512
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# second last output layer
#	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
#	d = InstanceNormalization(axis=-1)(d)
#	d = LeakyReLU(alpha=0.2)(d)
	# patch output
	patch_out = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	# define model
	model = Model(in_image, patch_out)
	# compile model
	model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
	return model

# generator a resnet block
def resnet_block(n_filters, input_layer):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# first layer convolutional layer
	g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(input_layer)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# second convolutional layer
	g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	# concatenate merge channel-wise with input layer
	g = Concatenate()([g, input_layer])
	return g

# define the standalone generator model
def define_generator(image_shape, n_resnet=6): #n_resnet=9
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=image_shape)
	# c7s1-64
	g = Conv2D(64, (7,7), padding='same', kernel_initializer=init)(in_image)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# d128
	g = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# d256
	g = Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# R256
	for _ in range(n_resnet):
		g = resnet_block(256, g)
	# u128
	g = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# u64
	g = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# c7s1-3
	g = Conv2D(3, (7,7), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	out_image = Activation('tanh')(g)
	# define model
	model = Model(in_image, out_image)
	return model

# define a composite model for updating generators by adversarial and cycle loss
def define_composite_model(g_model_1, d_model, g_model_2, image_shape):
	# ensure the model we're updating is trainable
	g_model_1.trainable = True
	# mark discriminator as not trainable
	d_model.trainable = False
	# mark other generator model as not trainable
	g_model_2.trainable = False
	# discriminator element
	input_gen = Input(shape=image_shape)
	gen1_out = g_model_1(input_gen)
	output_d = d_model(gen1_out)
	# identity element
	input_id = Input(shape=image_shape)
	output_id = g_model_1(input_id)
	# forward cycle
	output_f = g_model_2(gen1_out)
	# backward cycle
	gen2_out = g_model_2(input_id)
	output_b = g_model_1(gen2_out)
	# define model graph
	model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])
	# define optimization algorithm configuration
	opt = Adam(lr=0.0002, beta_1=0.5)
	# compile model with weighting of least squares loss and L1 loss
	model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=opt)#mse, mae, mae, mae
	return model

# select a batch of random samples, returns images and target
def generate_real_samples(ix, dataset, n_samples, patch_shape, mode = 'random'): #ix
    
    if mode == 'random':
        # choose random instances
        ix = randint(0, dataset.shape[0], n_samples)
        # retrieve selected images
        X = dataset[ix]
    else:
        X = dataset[ix]
        X = np.expand_dims(X, axis = 0)
        
    # generate ✬real✬ class labels (1)
    y = np.ones((n_samples, patch_shape, patch_shape, 1))
    return X, y

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, dataset, patch_shape):
    # generate fake instance
    X = g_model.predict(dataset)
    # create ✬fake✬ class labels (0)
    y = np.zeros((len(X), patch_shape, patch_shape, 1))
    return X, y

# update image pool for fake images
def update_image_pool(pool, images, max_size=50):
    selected = list()
    for image in images:
        if len(pool) < max_size:
            # stock the pool
            pool.append(image)
            selected.append(image)
        elif random() < 0.5:
            # use image, but don✬t add it to the pool
            selected.append(image)
        else:
            # replace an existing image and use replaced image
            ix = randint(0, len(pool))
            selected.append(pool[ix])
            pool[ix] = image        
    return np.asarray(selected)

# load and prepare training images
def load_real_samples(filename):
	# load compressed arrays
	data = np.load(filename)
	# unpack arrays
	X1, X2 = data['arr_0'], data['arr_1']
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]

# load metadata
def load_metadata(filename):
    # load compressed arrays
    data = np.load(filename)
    # unpack arrays
    M1, M2 = data['arr_0'], data['arr_1']
    return [M1, M2]

# save the generator models to file
def save_models(step, g_model_AtoB, g_model_BtoA):
    # save the first generator model
    filename1 = 'g_model_AtoB_%06d.h5' % (step+1)
    g_model_AtoB.save(filename1)
    # save the second generator model
    filename2 = 'g_model_BtoA_%06d.h5' % (step+1)
    g_model_BtoA.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))
       
# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, trainX, name, n_samples=5):
    # select a sample of input images
    X_in, _ = generate_real_samples(step, trainX, n_samples, 0, mode = 'random')
    # generate translated images
    X_out, _ = generate_fake_samples(g_model, X_in, 0)
    # scale all pixels from [-1,1] to [0,1]
    X_in = (X_in + 1) / 2.0
    X_out = (X_out + 1) / 2.0
    # plot real images
    for i in range(n_samples):
        plt.subplot(2, n_samples, 1 + i)
        plt.axis('off')
        plt.imshow(X_in[i])
    # plot translated image
    for i in range(n_samples):
        plt.subplot(2, n_samples, 1 + n_samples + i)
        plt.axis('off')
        plt.imshow(X_out[i])
    # save plot to file
    filename1 = '%s_generated_plot_%06d.png' % (name, (step+1))
    plt.savefig(filename1)
    plt.close()
    
# scale samples to [-1,1]
def scale_sample(X):
    sc_X = (X - 127.5) / 127.5
    return sc_X

# scale patches in the training data format
def scale_patches(patches):
    preprocessed_patches = []
    for i in range(len(patches)):
        p = scale_sample(patches[i])
        preprocessed_patches.append(p)
    return np.array(preprocessed_patches)

# scale each channel of image
def scale_channel(X):
    sc_X = np.zeros(X.shape, dtype= 'uint8')
    for i in np.arange(X.shape[2]):
        sc_X[:,:,i] = X[:,:,i]/ math.sqrt(np.mean(X[:,:,i]))
    return sc_X

# rescale samples to original format
def rescale_sample(X):
    sc_X = (X+1)/2.0
    sc_X = np.uint8(sc_X*255) 
    return sc_X

# predict a list of patches
def predict_patches(src_patch, model):
    tar_patch = []
    for i in range(len(src_patch)):
        # predict patch
        pat = model.predict(np.expand_dims(src_patch[i], axis= 0))
        # scale from [-1,1] to [0,1]
        pat = (pat + 1) / 2.0
        tar_patch.append(np.uint8(pat*255))
    return np.concatenate(tar_patch)

# custom metric
def total_loss(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred) + style_loss(y_true, y_pred) #+ perceptual_loss(y_true, y_pred)

# custom metric
def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)

# custom metric
def perceptual_loss(y_true, y_pred):
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=(256,256,3))
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False
    return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))

# custom metric
def mean_absolute_error(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)

# custom metric
def mean_square_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

# custom metric
def total_variation_loss(y_pred):
    assert K.ndim(y_pred) == 4
    if K.image_data_format() == 'channels_first':
        a = K.square(
            y_pred[:, :, :img_rows - 1, :img_cols - 1] - y_pred[:, :, 1:, :img_cols - 1])
        b = K.square(
            y_pred[:, :, :img_rows - 1, :img_cols - 1] - y_pred[:, :, :img_rows - 1, 1:])
    else:
        a = K.square(
            y_pred[:, :img_rows - 1, :img_cols - 1, :] - y_pred[:, 1:, :img_cols - 1, :])
        b = K.square(
            y_pred[:, :img_rows - 1, :img_cols - 1, :] - y_pred[:, :img_rows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

# custom metric
def gram_matrix(x):
    features = K.batch_flatten(x)
    gram = K.dot(features, K.transpose(features))
    return gram

# custom metric
def style_loss(style, combination):
    assert K.ndim(style) == 4
    assert K.ndim(combination) == 4
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_rows * img_cols
    return K.sum(K.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

# custom metric
def content_loss(base, combination):
    return K.sum(K.square(combination - base))

# preprocessing of source image
def flip_contrast(src_img):
    
    src_img[:,:,0] = 255 - src_img[:,:,0]
    src_img[:,:,1] = 255 - src_img[:,:,1]
    src_img[:,:,2] = 255 - src_img[:,:,2]
    return src_img

# preprocessing of source image
def normalize_intensity(src_img, method='max'):
    
    if method == 'max':
        src_img[:,:,0] = src_img[:,:,0]/np.max(src_img[:,:,0])
        src_img[:,:,1] = src_img[:,:,1]/np.max(src_img[:,:,1])
        src_img[:,:,2] = src_img[:,:,2]/np.max(src_img[:,:,2])
        
    elif method == 'mean':
        src_img[:,:,0] = src_img[:,:,0]/np.mean(src_img[:,:,0])
        src_img[:,:,1] = src_img[:,:,1]/np.mean(src_img[:,:,1])
        src_img[:,:,2] = src_img[:,:,2]/np.mean(src_img[:,:,2])
        
    else:
        src_img[:,:,0] = src_img[:,:,0]/np.sqrt(np.mean(src_img[:,:,0]))
        src_img[:,:,1] = src_img[:,:,1]/np.sqrt(np.mean(src_img[:,:,1]))
        src_img[:,:,2] = src_img[:,:,2]/np.sqrt(np.mean(src_img[:,:,2]))
        
    return src_img

# preprocessing of source image
def square_root_img(src_img):
    src_img_sqr = np.zeros((src_img.shape[:3]), dtype=np.float64)
    src_img_sqr[:,:,0] = np.sqrt(src_img[:,:,0])
    src_img_sqr[:,:,1] = np.sqrt(src_img[:,:,1])
    src_img_sqr[:,:,2] = np.sqrt(src_img[:,:,2])
    return src_img_sqr