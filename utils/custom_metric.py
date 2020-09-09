# -*- coding: utf-8 -*-
"""
Created on Wed May 23 11:20:06 2018

@author: si62qit
"""
from keras import backend as K

img_nrows = 256
img_ncols = 256


def mean_absolute_error(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)

def total_variation_loss(x):
    assert K.ndim(x) == 4
    if K.image_data_format() == 'channels_first':
        a = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, 1:, :img_ncols - 1])
        b = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, :img_nrows - 1, 1:])
    else:
        a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
        b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

def create_adv_loss(discriminator):
    def loss(y_true, y_pred):
        return K.log(1.0 - discriminator.predict(y_pred))
    return loss

def gen_loss(y_true, y_pred, discriminator):
    return (mean_absolute_error(y_true, y_pred) + 0.99 * total_variation_loss(y_pred) + 0.01 * (1.0 - discriminator.predict(y_pred)))