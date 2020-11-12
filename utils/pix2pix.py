# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 12:19:49 2019

@author: P.Pradhan

This script has Pix2Pix model.
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

# example of defining a composite model for training the generator model
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam, RMSprop
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.utils.vis_utils import plot_model

import numpy as np
import matplotlib.pyplot as plt
from numpy import load
from numpy import vstack
from numpy.random import randint
from imageProcessing.patches import image_to_patch, patch_to_image, filter_patches

# specify image size
img_rows = img_cols = 256

#%%
# rescale samples to original format
def scale_sample(X):
    sc_X = (X - 127.5) / 127.5
    return sc_X

# Scale patches in the training data format
def scale_patches(patches):
    preprocessed_patches = []
    for i in range(len(patches)):
        p = scale_sample(patches[i])
        preprocessed_patches.append(p)
        
    return np.array(preprocessed_patches)

# define the discriminator model
def define_discriminator(image_shape):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# source image input
	in_src_image = Input(shape=image_shape)
	# target image input
	in_target_image = Input(shape=image_shape)
	# concatenate images channel-wise
	merged = Concatenate()([in_src_image, in_target_image])
	# C64
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
	d = LeakyReLU(alpha=0.2)(d)
	# C128
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C256
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C512
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# second last output layer
#	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
#	d = BatchNormalization()(d)
#	d = LeakyReLU(alpha=0.2)(d)
	# patch output
	d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	patch_out = Activation('sigmoid')(d)
	# define model
	model = Model([in_src_image, in_target_image], patch_out)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])#0.5
	return model

# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add downsampling layer
	g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# conditionally add batch normalization
	if batchnorm:
		g = BatchNormalization()(g, training=True)
	# leaky relu activation
	g = LeakyReLU(alpha=0.2)(g)
	return g

# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add upsampling layer
	g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# add batch normalization
	g = BatchNormalization()(g, training=True)
	# conditionally add dropout
	if dropout:
		g = Dropout(0.5)(g, training=True)
	# merge with skip connection
	g = Concatenate()([g, skip_in])
	# relu activation
	g = Activation('relu')(g)
	return g

# define the standalone generator model
def define_generator(image_shape=(256,256,3)):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=image_shape)
	# encoder model: C64-C128-C256-C512-C512-C512-C512-C512
	e1 = define_encoder_block(in_image, 64, batchnorm=False)
	e2 = define_encoder_block(e1, 128)
	e3 = define_encoder_block(e2, 256)
	e4 = define_encoder_block(e3, 512)
	e5 = define_encoder_block(e4, 512)
	e6 = define_encoder_block(e5, 512)
	e7 = define_encoder_block(e6, 512)
	# bottleneck, no batch norm and relu
	b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
	b = Activation('relu')(b)
	# decoder model: CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
	d1 = decoder_block(b, e7, 512)
	d2 = decoder_block(d1, e6, 512)
	d3 = decoder_block(d2, e5, 512)
	d4 = decoder_block(d3, e4, 512, dropout=False)
	d5 = decoder_block(d4, e3, 256, dropout=False)
	d6 = decoder_block(d5, e2, 128, dropout=False)
	d7 = decoder_block(d6, e1, 64, dropout=False)
	# output
	g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
	out_image = Activation('tanh')(g)
	# define model
	model = Model(in_image, out_image)
	return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# define the source image
	in_src = Input(shape=image_shape)
	# connect the source image to the generator input
	gen_out = g_model(in_src)
	# connect the source input and generator output to the discriminator input
	dis_out = d_model([in_src, gen_out])
	# src image as input, generated image and classification output
	model = Model(in_src, [dis_out, gen_out])
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['mse', 'mae'], optimizer=opt, loss_weights=[1,10])#'mae'
	return model

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
    data = load(filename)
    # unpack arrays
    M1, M2 = data['arr_0'], data['arr_1']
    return [M1, M2]

# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
	# unpack dataset
	trainA, trainB = dataset
	# choose random instances
	ix = randint(0, trainA.shape[0], n_samples)
	# retrieve selected images
	X1, X2 = trainA[ix], trainB[ix]
	# generate 'real' class labels (1)
	y = np.ones((n_samples, patch_shape, patch_shape, 1))
	return [X1, X2], y

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
	# generate fake instance
	X = g_model.predict(samples)
	# create 'fake' class labels (0)
	y = np.zeros((len(X), patch_shape, patch_shape, 1))
	return X, y

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, dataset, n_samples=3):
	# select a sample of input images
	[X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
	# generate a batch of fake samples
	X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
	# scale all pixels from [-1,1] to [0,1]
	X_realA = (X_realA + 1) / 2.0
	X_realB = (X_realB + 1) / 2.0
	X_fakeB = (X_fakeB + 1) / 2.0
	# plot real source images
	for i in range(n_samples):
		plt.subplot(3, n_samples, 1 + i)
		plt.axis('off')
		plt.imshow(X_realA[i])
	# plot generated target image
	for i in range(n_samples):
		plt.subplot(3, n_samples, 1 + n_samples + i)
		plt.axis('off')
		plt.imshow(X_fakeB[i])
	# plot real target image
	for i in range(n_samples):
		plt.subplot(3, n_samples, 1 + n_samples*2 + i)
		plt.axis('off')
		plt.imshow(X_realB[i])
	# save plot to file
	filename1 = 'plot_%06d.png' % (step+1)
	plt.savefig(filename1)
	plt.close()
	# save the generator model
	filename2 = 'model_%06d.h5' % (step+1)
	g_model.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))

# create a line plot of loss for the gan and save to file
def plot_history(d1_hist, d2_hist, g_hist):
    plt.plot(d1_hist, label='dloss1')
    plt.plot(d2_hist, label='dloss2')
    plt.plot(g_hist, label='gloss')
    plt.legend()
    filename = 'plot_line_plot_loss.png'
    plt.savefig(filename)
    plt.close()
    print('Saved %s' % (filename))
    
# train pix2pix models
def train(d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=5, n_patch=16):
    # unpack dataset
    trainA, trainB = dataset
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # lists for storing loss, for plotting later
    d1_hist, d2_hist, g_hist = list(), list(), list()
    # manually enumerate epochs
    for i in range(n_steps):
        # select a batch of real samples
        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
        # generate a batch of fake samples
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        # update discriminator for real samples
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        # update discriminator for generated samples
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        # update the generator
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        # summarize performance
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
        # record history
        d1_hist.append(d_loss1)
        d2_hist.append(d_loss2)
        g_hist.append(g_loss)
        # summarize model performance
        if (i+1) % (bat_per_epo * 10) == 0:
            summarize_performance(i, g_model, dataset) 
    # create line plot of training history
    plot_history(d1_hist, d2_hist, g_hist)
    
# rescale samples to original format
def rescale_sample(X):
    sc_X = (X+1)/2.0
    sc_X = np.uint8(sc_X*255) 
    return sc_X

# plot source, generated and target patches
def plot_patch(src_img, gen_img, tar_img):
    images = vstack((src_img, gen_img, tar_img))
    # scale from [-1,1] to [0,1]
    images = (images + 1) / 2.0
    titles = ['Source patch', 'Generated patch', 'Expected patch']
    # plot images row by row
    plt.figure()
    for i in range(len(images)):
        # define subplot
        plt.subplot(1, 3, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(images[i])
        # show title
        plt.title(titles[i])
    plt.show()

# predict all patches 
def predict_patches(src_patch, model):
    tar_patch = []
    for i in range(len(src_patch)):
        # predict patch
        pat = model.predict(np.expand_dims(src_patch[i], axis= 0))
        # scale from [-1,1] to [0,1]
        pat = (pat + 1) / 2.0
        tar_patch.append(np.uint8(pat*255))
    return np.concatenate(tar_patch)

# plot source, generated and target patches
def plot_image(src_img, gen_img, tar_img):
    images = [src_img, gen_img, tar_img]
    titles = ['Source Image', 'Generated Image', 'Expected Image']
    # plot images row by row
    plt.figure()
    for i in range(len(images)):
        # define subplot
        plt.subplot(1, 3, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(images[i])
        # show title
        plt.title(titles[i])
    plt.show()

# reconstruct image from patches and plot a randomly selected patch and a whole image 
def reconstruct_stain_image(X1, X2, model, image_dim= (5120, 5120, 3), patch_dim= 256):
    # select random example patch
    ix = randint(0, len(X1), 1)
    src_patch, tar_patch = X1[ix], X2[ix]
    # generate image from source
    gen_patch = model.predict(src_patch)
    # plot all three patches
    plot_patch(src_patch, gen_patch, tar_patch)
    # predict all patches
    gen_patch = predict_patches(X1, model)
    # reconstruct generated image from predicted patches
    gen_image = patch_to_image(gen_patch, image_dim, patch_dim)
    # real target image 
    tar_image = patch_to_image(rescale_sample(X2), image_dim, patch_dim)
    # reconstruct source image 
    src_image = patch_to_image(rescale_sample(X1), image_dim, patch_dim)
    # plot all three images
    plot_image(src_image, gen_image, tar_image)

# custom metric
def total_loss(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred) + total_variation_loss(y_pred)

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
def normalize_intensity(src_img, method='max'):
    if method == 'max':
        src_img[:,:,0] = src_img[:,:,0]/np.max(src_img[:,:,0])
        src_img[:,:,1] = src_img[:,:,1]/np.max(src_img[:,:,1])
        src_img[:,:,2] = src_img[:,:,2]/np.max(src_img[:,:,2])
        
    elif method == 'mean':
        src_img[:,:,0] = src_img[:,:,0]/np.mean(src_img[:,:,0])
        src_img[:,:,1] = src_img[:,:,1]/np.mean(src_img[:,:,1])
        src_img[:,:,2] = src_img[:,:,2]/np.mean(src_img[:,:,2])
    return src_img

# preprocessing of source image
def square_root_img(src_img):
    src_img_sqr = np.zeros((src_img.shape[:3]), dtype=np.float64)
    src_img_sqr[:,:,0] = np.sqrt(src_img[:,:,0])
    src_img_sqr[:,:,1] = np.sqrt(src_img[:,:,1])
    src_img_sqr[:,:,2] = np.sqrt(src_img[:,:,2])
    return src_img_sqr

# preprocessing of source image
def flip_contrast(src_img):
    src_img[:,:,0] = 255 - src_img[:,:,0]
    src_img[:,:,1] = 255 - src_img[:,:,1]
    src_img[:,:,2] = 255 - src_img[:,:,2]
    return src_img

# preprocessing of source image
def filter_patch_pair(src_patch, tar_patch, min_mean=200, min_std=100):
    idx, fil_tar_patch = filter_patches(tar_patch, min_mean=min_mean, min_std=min_std)
    fil_src_patch = src_patch[idx]
    return fil_src_patch, fil_tar_patch

