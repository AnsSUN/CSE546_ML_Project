#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 22:50:08 2018

@author: anshul
"""
import time
from PIL import Image
import pandas as pd
import numpy as np
import scipy as sci
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
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
Y = pd.read_csv("Data_Entry_2017.csv", usecols= [1, 2])
Y= Y.values

ix = np.isin(X, results)
Y_train=Y[np.where(ix)]

# For 100 images to train the dataset and 25 images for test
final_results=[]
for size in [1000, 2000, 3000, 4000]:
    test_size=int(0.2*size) + size
    image_shape=[]
#X_img_val=[]
    images = Image.open(path_save+dirs[0]).convert('RGBA')
    arr = np.array(images)
    image_shape.append(arr.shape)
    flat_arr = arr.ravel()

# convert it to a matrix
    vector = np.matrix(flat_arr) /255.
    X_img_val=vector
    

    for j in dirs[1:size]:
    
        images = Image.open(path_save+j).convert('RGBA')
        arr = np.array(images)
        image_shape.append(arr.shape)
        flat_arr = arr.ravel()

# convert it to a matrix
        vector = np.matrix(flat_arr) /255.
        X_img_val=np.vstack((X_img_val, vector))
    

# Similarly for test images:
    test_image_shape=[]
#X_img_val=[]
    test_images = Image.open(path_save+dirs[size]).convert('RGBA')
    arr_test = np.array(test_images)
    test_image_shape.append(arr_test.shape)
    test_flat_arr = arr_test.ravel()

# convert it to a matrix
    vector_test = np.matrix(test_flat_arr) /255.
    X_test_img_val=vector_test
    

    for j in dirs[size:(test_size-1)]:
    
        test_images = Image.open(path_save+j).convert('RGBA')
        arr_test = np.array(test_images)
        test_image_shape.append(arr_test.shape)
        test_flat_arr = arr_test.ravel()

# convert it to a matrix
        vector_test = np.matrix(test_flat_arr) /255.
        X_test_img_val=np.vstack((X_test_img_val, vector_test))
    
    Y_test_img= Y_train[size:test_size]
# applying Principal component Analysis for diamensionality reduction:
    n_component=10
    pca = PCA(n_components=n_component, whiten=False)
    start_time= time.time()

    pca.fit(X_img_val)
    X_train_pca= pca.transform(X_img_val)

    end_time= time.time()
    time_taken= end_time-start_time
    print('Fit time elapsed: {}'.format(time_taken))

    pca

# Training the models on PCA - SVM semi supervised learning.
    X_test_pca =pca.transform(X_test_img_val)

    start_time= time.time()
    print('PCA diamensions for fitting the model: {}'.format(n_component))
    print('Size of samples for fitting the model: {}'.format(size))
    new_model = SVC()
    
    # condition to not go beyond original sample size:
    
    new_model.fit(X_train_pca, Y_train[0:len(X_train_pca)].ravel())
    Y_pred = new_model.predict(X_test_pca)
    score = accuracy_score(Y_test_img, Y_pred)
    
    end_time= time.time()
    time_taken= end_time-start_time
    print(' * Accuracy: %.1f %%' % (100. * score))
    print(' * Fit time elapsed: %.1fs' % time_taken)
    final_results.append({'PCA dim:': n_component, 'sample_size': size, 'accuracy': score, 'time': time_taken})
    
   
df = pd.DataFrame(final_results)

df.to_csv('pca_results.csv', index=False)

# reform a numpy array of the original shape
#arr2 = np.asarray(vector*255.).reshape(image_shape[])
    
# Training woth PCA:
    
