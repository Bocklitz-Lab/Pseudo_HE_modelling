# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 16:56:19 2020

@author: si62qit
"""

import warnings
warnings.filterwarnings("ignore")
import glob
import ntpath
import csv
from skimage.io import imread

#%%

"""
Run this section to make csv file
"""

imageD = 'C:/Users/si62qit/Documents/PhDJenaPranita/codesPython/HEstaining_GAN/2_BackgroundMask/*.png'

file_name_list = []
for filename in glob.glob(imageD): #assuming gif
    im = imread(filename)
    file_name_list.append(ntpath.basename(filename))
    
with open('data1.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerow( ('MaskID_1', 'MaskID_2') )
    for i in range(0,38,2):
        writer.writerow( ('/2_BackgroundMask/'+ file_name_list[i], '/2_BackgroundMask/'+ file_name_list[i+1] ) )
    f.close()