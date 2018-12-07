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
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
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
size =4000
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
for n_component in [10, 20, 50, 100, 200, 500]:

    for sample_size in [1000, 2000, 3000, 4000]:
        pca = PCA(n_components=n_component, whiten=False)
        start_time= time.time()

        pca.fit(X_img_val[0:sample_size])
        X_train_pca= pca.transform(X_img_val[0:sample_size])

        end_time= time.time()
        time_taken= end_time-start_time
        print('Fit time elapsed: {}'.format(time_taken))

        pca
        test_size_new=int(0.2*sample_size) + sample_size
# Training the models on PCA - SVM semi supervised learning.
        X_test_pca =pca.transform(X_test_img_val[0: (test_size_new-sample_size)])
        start_time= time.time()
        print('PCA diamensions for fitting the model: {}'.format(n_component))
        print('Size of samples for fitting the model: {}'.format(sample_size))
        new_model = SVC()
    
    # Creating new samples with less size:
        
        Y_test_img_new= Y_test_img[0: (test_size_new-sample_size)]
        
        #X_img_val_split=X_train_pca[0: sample_size]

# Similarly for test images:
    
        #X_test_img_val_split=X_test_img_val[sample_size: test_size]
    
        #Y_test_img_split= Y_train[sample_size:test_size]
        #X_pca = pca.transform(X_img_val_split)
    ###################################################################################
        new_model.fit(X_train_pca[0:sample_size], Y_train[0: sample_size].ravel())
        Y_pred = new_model.predict(X_test_pca)
        score = accuracy_score(Y_test_img_new, Y_pred)
    
        end_time= time.time()
        time_taken= end_time-start_time
        print("Classification report for classifier %s:\n%s\n"
      % (new_model, metrics.classification_report(Y_test_img_new, Y_pred)))
        print(' * Accuracy: %.1f %%' % (100. * score))
        print(' * Fit time elapsed: %.1fs' % time_taken)
        final_results.append({'PCA dim:': n_component, 'sample_size': sample_size, 'accuracy': score, 'time': time_taken, 'diseases classify': (new_model, metrics.classification_report(Y_test_img_new, Y_pred))})
    
   
df = pd.DataFrame(final_results)

df.to_csv('pca_results.csv', index=False)

# reform a numpy array of the original shape
#arr2 = np.asarray(vector*255.).reshape(image_shape[]

"""
#for plot
pca_data =np.array([10, 20, 50, 100, 200, 500])
sample_data= np.matrix([1000, 2000, 3000, 4000]) 
accuracies= np.matrix([[0.7, 0.7, 0.7, 0.7, 0.7, 0.7], 
             [0.635, 0.635, 0.635, 0.635, 0.635, 0.635],
             [0.6, 0.64, 0.643, 0.6433, 0.64333, 0.643333],
             [0.5, 0.59, 0.598, 0.5987, 0.59875, 0.59875]])

plt.plot(sample_data.T, accuracies[:,0])
plt.plot(sample_data.T, accuracies[:,1])
plt.plot(sample_data.T, accuracies[:,2])
plt.plot(sample_data.T, accuracies[:,3])
plt.plot(sample_data.T, accuracies[:,4])
plt.plot(sample_data.T, accuracies[:,5])
plt.title("ROC AUC Curve")
plt.xlabel("sample data")
plt.ylabel("Score")
plt.legend(pca_data)

"""

