#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 22:50:08 2018

@author: anshul
"""

from PIL import Image
import pandas as pd
import numpy as np
import scipy as sci
import os, sys


def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

#path = "/home/anshul/MAchine_Learning_code/Project/Input_img/"
path_save = "/home/anshul/MAchine_Learning_code/Project/Converted/"
dirs = os.listdir( path_save )
results =[]
for item in dirs:
    if os.path.isfile(path_save+item):
        results.append(item)
        


X= pd.read_csv("Data_Entry_2017.csv", usecols= [0])
X= X.values
Y = pd.read_csv("Data_Entry_2017.csv", usecols= [1])
Y= Y.values

ix = np.isin(X, results)
Y_train=Y[np.where(ix)]

# For 100 images to train the dataset and 25 images for test
image_shape=[]
#X_img_val=[]
images = Image.open(path_save+dirs[0]).convert('RGBA')
arr = np.array(images)
image_shape.append(arr.shape)
flat_arr = arr.ravel()

# convert it to a matrix
vector = np.matrix(flat_arr) /255.
X_img_val=vector.T
    

for j in dirs[1:100]:
    
    images = Image.open(path_save+j).convert('RGBA')
    arr = np.array(images)
    image_shape.append(arr.shape)
    flat_arr = arr.ravel()

# convert it to a matrix
    vector = np.matrix(flat_arr) /255.
    X_img_val=np.hstack((X_img_val, vector.T))
    

# Similarly for test images:
test_image_shape=[]
#X_img_val=[]
test_images = Image.open(path_save+dirs[100]).convert('RGBA')
arr_test = np.array(test_images)
test_image_shape.append(arr_test.shape)
test_flat_arr = arr_test.ravel()

# convert it to a matrix
vector_test = np.matrix(test_flat_arr) /255.
X_test_img_val=vector_test.T
    

for j in dirs[101:125]:
    
    test_images = Image.open(path_save+j).convert('RGBA')
    arr_test = np.array(test_images)
    test_image_shape.append(arr_test.shape)
    test_flat_arr = arr_test.ravel()

# convert it to a matrix
    vector_test = np.matrix(test_flat_arr) /255.
    X_test_img_val=np.hstack((X_test_img_val, vector_test.T))
    

# reform a numpy array of the original shape
#arr2 = np.asarray(vector*255.).reshape(image_shape[])
    
# Training woth PCA:
    
