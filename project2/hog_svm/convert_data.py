
# coding: utf-8

# In[1]:

from sklearn import svm
import numpy as np
from sklearn.metrics import classification
from numpy import linalg as LA
import my_io
import logging
from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt
import ConfigParser
import datetime
import os
import csv
import scipy.io
import cPickle
import gzip
import os
import sys
import timeit
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
import numpy

import theano
import theano.tensor as T


# In[2]:

mat_file_path = './train/train.mat'

print 'load mat file'
mat = scipy.io.loadmat(mat_file_path)
all_data = mat['train']
print 'mat file read finish, setting input and output'
n = len(all_data['y'][0][0])

train_set_y = numpy.arange(n, dtype = int)
for i in range(0, n):
    #train_set_y[i] = all_data['y'][0][0][i][0] - 1
    train_set_y[i] = all_data['y'][0][0][i][0] 


# In[3]:

train_set_x = all_data['X_cnn'][0][0].reshape(n, 36865)
train_set_x = train_set_x[:,:36864]
del all_data


# In[4]:

print train_set_x.shape
print train_set_y.shape
type(train_set_x)


# In[5]:

from split_data import splitData


# In[6]:

from my_io import startLog


# In[7]:

X_test, X_train, y_train, y_test = splitData(train_set_x, train_set_y, 0.2, 1)
print(y_test)
print(y_train)
import numpy as np

np.savetxt('X_test.csv',X_test,delimiter=',')
np.savetxt('X_train.csv',X_train,delimiter=',')
np.savetxt('y_test.csv',y_test,delimiter=',')
np.savetxt('y_train.csv',y_train,delimiter=',')

# In[8]:



# In[14]:
"""
predict_probs = svc.predict_proba(X_test)

predict = my_io.toZeroOne(predict_probs)
# error = classification.zero_one_loss(y_test, predict)
loss = np.subtract(predict,y_test)


# In[ ]:

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA


# In[ ]:

classif = OneVsRestClassifier(SVC(kernel='linear'))
classif.fit(X_train, y_train)


# In[ ]:



logger.info('start predict')
predict_probs = svc.predict_proba(X_test)

predict = my_io.toZeroOne(predict_probs)
# error = classification.zero_one_loss(y_test, predict)
loss = np.subtract(predict,y_test)

error = LA.norm(loss)
logger.info('zero one loss %f',error)


# In[ ]:

"""

