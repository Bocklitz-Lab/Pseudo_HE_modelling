# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:04:02 2020

@author: P.Pradhan

This script is to generate subplot for qualitative evaluation in the manuscript.

"""

from skimage.io import imread
import os
import pandas as pd
import ntpath
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

path = os.getcwd()
df = pd.read_csv(path + '/6_Pix2Pix_vs_cycleGAN/qualitative_evaluation.csv')

#%%
fig, ax = plt.subplots(5, 6, figsize=(10, 6))
fig.subplots_adjust(hspace=None, wspace=None)
scalebar = ScaleBar(0.22, 'um', length_fraction = 0.5) # 1 pixel = 0.22 meter

ax[0,0].set_title('(A)')
ax[0,1].set_title('(B)')
ax[0,2].set_title('(C)')
ax[0,3].set_title('(D)')
ax[0,4].set_title('(E)')
ax[0,5].set_title('(F)')

ax[0,0].set_ylabel('Image 1')
ax[1,0].set_ylabel('Image 2')
ax[2,0].set_ylabel('Image 3')
ax[3,0].set_ylabel('Image 4')
ax[4,0].set_ylabel('Image 5')

for i, item in df.iterrows():
    if i >= len(df):
        break
    mm = imread(path + item[0])
    he = imread(path + item[6])
    pix2pix = imread(path + "/6_Pix2Pix_vs_cycleGAN/pix2pix/results/" + ntpath.basename(item[2]))
    cyclegan_post = imread(path + "/6_Pix2Pix_vs_cycleGAN/cycleGAN/results/postProcess/01_reduce_contrast/" + ntpath.basename(item[2]))
    cyclegan = imread(path + "/6_Pix2Pix_vs_cycleGAN/cycleGAN/results/" + ntpath.basename(item[2]))
    cyclegan_rec = imread(path + "/6_Pix2Pix_vs_cycleGAN/cycleGAN/reconstructed/" + ntpath.basename(item[2]))
    scalebar = ScaleBar(0.22, 'um')
    
    ax[i,0].imshow(mm[item[9]:item[10], item[11]:item[12]])
    ax[i,0].add_artist(scalebar)
    plt.imsave('C:/Users/si62qit/Documents/PhDJenaPranita/pseudoHE/images/results/C0_R'+str(i), mm[item[9]:item[10], item[11]:item[12]])
    
    ax[i,1].imshow(he[item[9]:item[10], item[11]:item[12]])
    plt.imsave('C:/Users/si62qit/Documents/PhDJenaPranita/pseudoHE/images/results/C1_R'+str(i), he[item[9]:item[10], item[11]:item[12]])
    
    ax[i,2].imshow(pix2pix[item[9]:item[10], item[11]:item[12]])
    plt.imsave('C:/Users/si62qit/Documents/PhDJenaPranita/pseudoHE/images/results/C2_R'+str(i), pix2pix[item[9]:item[10], item[11]:item[12]])
    
    ax[i,3].imshow(cyclegan[item[9]:item[10], item[11]:item[12]])
    plt.imsave('C:/Users/si62qit/Documents/PhDJenaPranita/pseudoHE/images/results/C3_R'+str(i), cyclegan[item[9]:item[10], item[11]:item[12]])
    
    ax[i,4].imshow(cyclegan_post[item[9]:item[10], item[11]:item[12]])
    plt.imsave('C:/Users/si62qit/Documents/PhDJenaPranita/pseudoHE/images/results/C4_R'+str(i), cyclegan_post[item[9]:item[10], item[11]:item[12]])
    
    ax[i,5].imshow(cyclegan_rec[item[9]:item[10], item[11]:item[12]])
    plt.imsave('C:/Users/si62qit/Documents/PhDJenaPranita/pseudoHE/images/results/C5_R'+str(i), cyclegan_rec[item[9]:item[10], item[11]:item[12]])
    
    plt.setp(ax[i,0].get_xticklabels(), visible=False)
    plt.setp(ax[i,0].get_yticklabels(), visible=False)
    ax[i,0].tick_params(axis='both', which='both', length=0)
#    ax[i,0].set_axis_off()
    ax[i,0].set_aspect('equal')
    
    plt.setp(ax[i,1].get_xticklabels(), visible=False)
    plt.setp(ax[i,1].get_yticklabels(), visible=False)
    ax[i,1].tick_params(axis='both', which='both', length=0)
#    ax[i,1].set_axis_off()
    ax[i,1].set_aspect('equal')
    
    plt.setp(ax[i,2].get_xticklabels(), visible=False)
    plt.setp(ax[i,2].get_yticklabels(), visible=False)
    ax[i,2].tick_params(axis='both', which='both', length=0)
#    ax[i,2].set_axis_off()
    ax[i,2].set_aspect('equal')
    
    plt.setp(ax[i,3].get_xticklabels(), visible=False)
    plt.setp(ax[i,3].get_yticklabels(), visible=False)
    ax[i,3].tick_params(axis='both', which='both', length=0)
#    ax[i,3].set_axis_off()
    ax[i,3].set_aspect('equal')
    
    plt.setp(ax[i,4].get_xticklabels(), visible=False)
    plt.setp(ax[i,4].get_yticklabels(), visible=False)
    ax[i,4].tick_params(axis='both', which='both', length=0)
#    ax[i,3].set_axis_off()
    ax[i,4].set_aspect('equal')
    
    plt.setp(ax[i,5].get_xticklabels(), visible=False)
    plt.setp(ax[i,5].get_yticklabels(), visible=False)
    ax[i,5].tick_params(axis='both', which='both', length=0)
#    ax[i,3].set_axis_off()
    ax[i,5].set_aspect('equal')
        
fig.tight_layout()
plt.savefig("C:/Users/si62qit/Documents/PhDJenaPranita/pseudoHE/images/results/qualitative_evaluation.png", bbox_inches='tight', dpi=500)

#%% Save the cropped patches seperately in a folder.

for i, item in df.iterrows():
    if i == 1:
        mm = imread(path + item[0])
        he = imread(path + item[6])
        mm_inv = imread(path + '/3_GeneratedHE_cycleGAN_inverted_MM/inv_MM/' + ntpath.basename(item[6]))
        cyclegan_inv = imread(path + "/3_GeneratedHE_cycleGAN_patch/results/" + ntpath.basename(item[2]))
        cyclegan_act = imread(path + "/6_Pix2Pix_vs_cycleGAN/cycleGAN/results/" + ntpath.basename(item[2]))
        
        plt.imsave('C:/Users/si62qit/Documents/PhDJenaPranita/pseudoHE/images/results/mm', mm[item[9]:item[10], item[11]:item[12]])
        plt.imsave('C:/Users/si62qit/Documents/PhDJenaPranita/pseudoHE/images/results/mm_inv', mm_inv[item[9]:item[10], item[11]:item[12]])
        plt.imsave('C:/Users/si62qit/Documents/PhDJenaPranita/pseudoHE/images/results/he', he[item[9]:item[10], item[11]:item[12]])
        plt.imsave('C:/Users/si62qit/Documents/PhDJenaPranita/pseudoHE/images/results/cyclegan', cyclegan_act[item[9]:item[10], item[11]:item[12]])
        plt.imsave('C:/Users/si62qit/Documents/PhDJenaPranita/pseudoHE/images/results/cyclegan_inv', cyclegan_inv[item[9]:item[10], item[11]:item[12]])