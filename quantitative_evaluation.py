# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 13:34:16 2020

@author: P.Pradhan

This script is used to make quantitative evaluation of the generated H&E images.
The quantitative evaluation is based on MSE, CSS and SSIM.

"""

# import all pacakages
from skimage.io import imread
import os
import pandas as pd
import numpy as np
import math
import ntpath
from skimage.measure import compare_ssim as ssim
from skimage.color import rgb2lab
import matplotlib.pyplot as plt

# get data directory
path = "C:/Users/si62qit/Documents/PhDJenaPranita/pseudoHE/"

# get .csv file in the data directory
df = pd.read_csv(os.getcwd() + '/data.csv')

def mse(imageA, imageB):
	""" The 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
    Args: 
        imageA, imageB (numpy array): two images for which MSE metric 
        needs to be calculated.
    Returns:
        err (float): MSE value  
    """
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def euclidean_distance(x,y):
    """
    This function calculates Euclidean distance between two vectors.
    Args:
        x,y (numpy array): Two vectors for which Euclidean distance needs to be 
        calculated.
    Returns:
        distance (int): Euclidean distance value.   
    """
    distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))
    return distance

def calculateDistance(x,y):
    """
    This function calculates RMSE distance between two vectors.
    Args:
        x,y (numpy array): Two vectors for which RMSE distance needs to be 
        calculated.
    Returns:
        distance (int): RMSE value.   
    """
    return np.sum((x-y)**2)

def sim(x_lab, y_lab):
    """
    This is utility function for calculating color_space_similarity index.
    The formula for sim is give in the manuscript.
    Args:
        x_lab, y_lab (numpy array): Two image vectors in LAB space.
    Returns:
        sim (float): SIM value calculated using formula.
    """
    dist_a = abs(x_lab[:,:,1].ravel()-y_lab[:,:,1].ravel())
    dist_b = abs(x_lab[:,:,2].ravel()-y_lab[:,:,2].ravel())
    
    if (np.max(dist_a)==0 and np.max(dist_b)==0):
        dist_a = 1-dist_a
        dist_b = 1-dist_b
    else:
        dist_a = 1-(dist_a/np.max(dist_a))
        dist_b = 1-(dist_b/np.max(dist_b))
        
    return np.concatenate((dist_a, dist_b), axis = 0)

def color_space_similarity(x, y, threshold=0.5):
    """
    This function is used to calculate CSS metric between two images.
    Args:
        Args:
        x,y (numpy array): Two vectors for which CSS needs to be calculated.
    Returns:
        CSS value.   
    """
    x_lab = rgb2lab(x)
    y_lab = rgb2lab(y)
    sim_values = sim(x_lab, y_lab)
    sim_values = [xx if xx > threshold else 0 for xx in sim_values]
    return np.mean(sim_values)

#%% Run this section to caluate all metrics between computational H&E image and 
# histopathological H&E image.
    
mse_none = []
mse_pix2pix = []
mse_cyclegan = []

ssim_none = []
ssim_pix2pix = []
ssim_cyclegan = []

css_none = []
css_pix2pix = []
css_cyclegan = []

for i, item in df.iterrows():
    if i >= len(df):
        break
    mm = imread(path + item[0])
    he = imread(path + item[6])
    mask = imread(path + item[7])
    pix2pix = imread(path + "/6_Pix2Pix_vs_cycleGAN/pix2pix/results/" + ntpath.basename(item[2]))
    cyclegan = imread(path + "/6_Pix2Pix_vs_cycleGAN/cycleGAN/results/" + ntpath.basename(item[2]))
    
    pix2pix = pix2pix[:,:,0:3]
    cyclegan = cyclegan[:,:,0:3]

    he[mask == 0] = 0
    pix2pix[mask == 0] = 0
    cyclegan[mask == 0] = 0
    
    mse_none.append(mse(he, he))
    ssim_none.append(ssim(he, he, data_range = he.max() - he.min(), multichannel=True))
    css_none.append(color_space_similarity(he, he, threshold=0.5))
    
    mse_pix2pix.append(mse(he, pix2pix))
    ssim_pix2pix.append(ssim(he, pix2pix, data_range = pix2pix.max() - pix2pix.min(), multichannel=True))
    css_pix2pix.append(color_space_similarity(he, pix2pix, threshold=0.5))
    
    mse_cyclegan.append(mse(he, cyclegan))
    ssim_cyclegan.append(ssim(he, cyclegan, data_range = cyclegan.max() - cyclegan.min(), multichannel=True))
    css_cyclegan.append(color_space_similarity(he, cyclegan, threshold=0.5))

print('Pathological H&E --> Average MSE: {%.2f}, Average SSIM: {%.2f}, Average CSS: {%.2f}' % (np.mean(mse_none), np.mean(ssim_none), np.mean(css_none)))
print('Pix2Pix --> Average MSE: {%.2f}, Average SSIM: {%.2f}, Average CSS: {%.2f}' % (np.mean(mse_pix2pix), np.mean(ssim_pix2pix), np.mean(css_pix2pix)))
print('CycleGAN --> Average MSE: {%.2f}, Average SSIM: {%.2f}, Average CSS: {%.2f}' % (np.mean(mse_cyclegan), np.mean(ssim_cyclegan), np.mean(css_cyclegan)))

#%% Save all metric values in csv file

df['Pix2Pix_MSE'] = mse_pix2pix 
df['Pix2Pix_SSIM'] = ssim_pix2pix
df['Pix2Pix_CSS'] = css_pix2pix

df['CycleGAN_MSE'] = mse_cyclegan 
df['CycleGAN_SSIM'] = ssim_cyclegan 
df['CycleGAN_CSS'] = css_cyclegan 

df.to_csv(path+'/6_Pix2Pix_vs_cycleGAN/quantitative_evaluation_test.csv', index=False)

#%% make boxplot of all metric values

from pylab import plot, show, savefig, xlim, figure, \
                hold, ylim, legend, boxplot, setp, axes

# function for setting the colors of the box plots pairs
def setBoxColors(bp):
    setp(bp['boxes'][0], color='blue')
    setp(bp['caps'][0], color='blue')
    setp(bp['caps'][1], color='blue')
    setp(bp['whiskers'][0], color='blue')
    setp(bp['whiskers'][1], color='blue')
    setp(bp['medians'][0], color='blue')

    setp(bp['boxes'][1], color='red')
    setp(bp['caps'][2], color='red')
    setp(bp['caps'][3], color='red')
    setp(bp['whiskers'][2], color='red')
    setp(bp['whiskers'][3], color='red')
    setp(bp['medians'][1], color='red')
    

# Some fake data to plot
A = [mse_pix2pix, ssim_pix2pix, css_pix2pix]
B = [mse_cyclegan, ssim_cyclegan, css_cyclegan]

fig, ([ax1, ax2, ax3]) = plt.subplots(1,3, figsize=(10,4))
hold(True)

# first boxplot pair
bp = ax1.boxplot([mse_pix2pix, mse_cyclegan], positions = [1, 2], widths = 0.6)
setBoxColors(bp)

# second boxplot pair
bp = ax2.boxplot([ssim_pix2pix, ssim_cyclegan], positions = [1, 2], widths = 0.6)
setBoxColors(bp)

# third boxplot pair
bp = ax3.boxplot([css_pix2pix, css_cyclegan], positions = [1, 2], widths = 0.6)
setBoxColors(bp)

# set axes limits and labels
ax1.set_xticklabels(['Pix2Pix', 'CycleGAN'])
ax2.set_xticklabels(['Pix2Pix', 'CycleGAN'])
ax3.set_xticklabels(['Pix2Pix', 'CycleGAN'])

ax1.set_title("MSE")
ax2.set_title("SSIM")
ax3.set_title("CSS")

savefig(path+'/6_Pix2Pix_vs_cycleGAN/boxcompare.png', dpi=300)
show()