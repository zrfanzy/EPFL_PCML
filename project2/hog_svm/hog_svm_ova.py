
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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
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
    train_set_y[i] = all_data['y'][0][0][i][0] - 1


# In[3]:

train_set_x = all_data['X_hog'][0][0].reshape(n, 5408)
del all_data


# In[4]:

print train_set_x.shape
type(train_set_x)


# In[5]:

from split_data import splitData


# In[6]:

from my_io import startLog


# In[7]:

X_test, X_train, y_train, y_test = splitData(train_set_x, train_set_y, 0.2, 1)


# In[8]:

my_io.startLog(__name__)
logger = logging.getLogger(__name__)
logger.info('msg %d %s' % (2015, 'test'))

logger.info('init svm classifier')
# svc = svm.SVC(probability = True)
logger.info('fitting svc')


# In[9]:

X_train[1].shape


# In[11]:

logger.info('size X_train:')
# svc = OneVsRestClassifier(SVC(random_state=0,decision_function_shape='ovr',kernel='rbf', verbose = True))
# svc = OneVsRestClassifier(LinearSVC(random_state=0,decision_function_shape='ovr',kernel='rbf', verbose = True))
# svc = LinearSVC(loss = 'hinge',verbose = True,)
svc = SVC(decision_function_shape = 'ovr',kernel='linear',verbose = True,)

svc.fit(X_train, y_train)


logger.info('training done start predicting')

# In[18]:

type(svc)
score =  svc.score(X_test, y_test)
logger.info(score)


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

