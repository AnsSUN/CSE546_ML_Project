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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.svm.libsvm import predict_proba
from sklearn.metrics import confusion_matrix
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
        


XY= pd.read_csv("Data_Entry_2017.csv", usecols= [0, 1])
XY= XY.values
#Y = pd.read_csv("Data_Entry_2017.csv", usecols= [1])
#Y= Y.values
Y_train_new=[]
ix = np.isin(XY[:, 0], results)
#Y_train=XY[np.where(ix), 1].T
XY_train_val= XY[np.where(ix)]
for items in results:
    ix_item = np.isin(XY_train_val[:, 0], items)
    Y_new=(XY_train_val[np.where(ix_item), 1])
    Y_train_new.append(Y_new)
    
Y_train_new = np.asarray(Y_train_new)
Y_train_new = Y_train_new[:, 0, 0]
df1 = pd.DataFrame(Y_train_new)

df1.to_csv('diseases.csv', index=False)

# For 100 images to train the dataset and 25 images for test

final_results=[]
shape_mat_inp= np.shape(results)
size =shape_mat_inp[0]
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
    
"""
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
    
Y_test_img= Y_train_new[size:test_size]
"""

# split into a training and testing set
#X_train, X_test, y_train, y_test = train_test_split(X_img_val, Y_train_new, test_size=0.20, random_state=42)
# applying Principal component Analysis for diamensionality reduction:
save_auc_roc_score = []
sample_size=2422

for n_component in [10, 20, 50, 100, 500, 600]:
    X_train, X_test, y_train, y_test = train_test_split(X_img_val, Y_train_new, test_size=0.20, random_state=42)
    for sample_size in [2422]:
        pca = PCA(n_components=n_component, svd_solver='randomized', whiten=True).fit(X_train)
        start_time= time.time()

        #pca.fit(X_img_val[0:sample_size])
        X_train_pca= pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
        end_time= time.time()
        time_taken= end_time-start_time
        print('Fit time elapsed: {}'.format(time_taken))

        pca
        #test_size_new=int(0.2*sample_size) + sample_size
# Training the models on PCA - SVM semi supervised learning.
        #X_test_pca =pca.transform(X_test_img_val[0: (test_size_new-sample_size)])
        #start_time= time.time()
        print('PCA diamensions for fitting the model: {}'.format(n_component))
        #print('Size of samples for fitting the model: {}'.format(sample_size))
        
        # SVM classification model:
        #parameters = {'C': [1e3, 5e3, 1e4, 5e4, 1e5], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
        #new_model = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), parameters, cv=5)
        new_model = SVC(kernel='rbf', probability=True, decision_function_shape='ovo')
    
    # Creating new samples with less size:
        
        #Y_test_img_new= Y_test_img[0: (test_size_new-sample_size)]
        
        #X_img_val_split=X_train_pca[0: sample_size]

# Similarly for test images:
    
        #X_test_img_val_split=X_test_img_val[sample_size: test_size]
    
        #Y_test_img_split= Y_train_new[sample_size:test_size]
        #X_pca = pca.transform(X_img_val_split)
    ###################################################################################
        new_model.fit(X_train_pca, y_train)
        Y_pred = new_model.predict(X_test_pca)
        Y_pred_prob= new_model.predict_log_proba(X_test_pca)
        confus_mat = confusion_matrix(y_test, Y_pred)
        ########################################
        #RAS=roc_auc_score(y_test, Y_pred_prob)
        #save_auc_roc_score.append(RAS)
        score = accuracy_score(y_test, Y_pred)
    
        end_time= time.time()
        time_taken= end_time-start_time
        print("Classification report for classifier %s:\n%s\n"
      % (new_model, metrics.classification_report(y_test, Y_pred)))
        print(' * Accuracy: %.1f %%' % (100. * score))
        #print('\n Roc-Auc Score: \n', RAS)
        print(' * Fit time elapsed: %.1fs' % time_taken)
        final_results.append({'PCA dim:': n_component, 'sample_size': sample_size, 'accuracy': score, 'time': time_taken, 'diseases classify': (new_model, metrics.classification_report(y_test, Y_pred))})
    
   
df = pd.DataFrame(final_results)

df.to_csv('pca_results.csv', index=False)

# reform a numpy array of the original shape
#arr2 = np.asarray(vector*255.).reshape(image_shape[]
"""
#for plot
pca_data =np.array([10, 20, 50, 100, 200, 500, 1000, 1500, 2000, 2500])
sample_data= np.matrix([10000, 20000, 30000, 40000]) 
accuracies= np.matrix([[0.492, 0.492, 0.491, 0.4915, 0.492, 0.492, 0.492, 0.492, 0.492, 0.492], 
             [0.513, 0.5135, 0.513, 0.51325, 0.51325, 0.51325, 0.51325, 0.51325, 0.51325, 0.51325],
             [0.51983, 0.52, 0.519167, 0.5195, 0.520167, 0.520167, 0.520167, 0.520167, 0.520167, 0.520167],
             [0.509, 0.5091, 0.5085, 0.509125, 0.509375, 0.509375, 0.509375, 0.509375, 0.509375, 0.509375]])

plt.plot(sample_data.T, accuracies[:,0])
plt.plot(sample_data.T, accuracies[:,1])
plt.plot(sample_data.T, accuracies[:,2])
plt.plot(sample_data.T, accuracies[:,3])
plt.plot(sample_data.T, accuracies[:,4])
plt.plot(sample_data.T, accuracies[:,5])
plt.plot(sample_data.T, accuracies[:,6])
plt.plot(sample_data.T, accuracies[:,7])
plt.plot(sample_data.T, accuracies[:,8])
plt.plot(sample_data.T, accuracies[:,9])

plt.title("Variations of accuracy with respect to samples")
plt.xlabel("sample data")
plt.ylabel("Score")
plt.legend(pca_data)

"""