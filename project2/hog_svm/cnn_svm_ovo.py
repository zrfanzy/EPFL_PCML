
# coding: utf-8
# source ~/python27-gordon/bin/activate
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

import theano.tensor as T



ft = 'cnn' #hog
Xtrain_file_path = '../train/'+ft+'_Xtrain.mat'
ytrain_file_path = '../train/'+ft+'_ytrain.mat'

print 'load mat file'+Xtrain_file_path
print 'load mat file'+ytrain_file_path

Xtrainmat = scipy.io.loadmat(Xtrain_file_path)
ytrainmat = scipy.io.loadmat(ytrain_file_path)

Xdata =Xtrainmat[ft+'_Xtrain']
ydata =ytrainmat[ft+'_ytrain']

print 'mat file read finish, setting input and output'
# n = len(all_data['y'][0][0])
n = 4800

y_train = numpy.arange(n, dtype = int)
for i in range(0, n):
#    train_set_y[i] = all_data['y'][0][0][i][0] - 1
    y_train[i] = ydata[i] - 1

# train_set_x = all_data['X_cnn'][0][0].reshape(n, 36865)
X_train = Xdata[:,:]



print X_train.shape
type(X_train)
#####
Xtest_file_path = '../train/'+ft+'_Xtest.mat'
ytest_file_path = '../train/'+ft+'_ytest.mat'

print 'load mat file'+Xtest_file_path
print 'load mat file'+ytest_file_path

Xtestmat = scipy.io.loadmat(Xtest_file_path)
ytestmat = scipy.io.loadmat(ytest_file_path)

Xdata =Xtestmat[ft+'_Xtest']
ydata =ytestmat[ft+'_ytest']

print 'mat file read finish, setting input and output'
# n = len(all_data['y'][0][0])
n = 1200

y_test = numpy.arange(n, dtype = int)
for i in range(0, n):
#    train_set_y[i] = all_data['y'][0][0][i][0] - 1
     y_test[i] = ydata[i] - 1

# train_set_x = all_data['X_cnn'][0][0].reshape(n, 36865)
X_test = Xdata[:,:]
print X_test.shape
type(X_test)


from my_io import startLog

# X_test, X_train, y_train, y_test = splitData(train_set_x, train_set_y, 0.2, 1)

my_io.startLog(__name__)
logger = logging.getLogger(__name__)
logger.info('msg %d %s' % (2015, 'test'))

logger.info('init svm classifier')

# svc = svm.SVC(probability = True)
logger.info('fitting svc, ovo RBF, feature')

logger.info(ft)


print X_train[1].shape

logger.info('size X_train:')
# svc = OneVsOneClassifier(SVC(random_state=0,decision_function_shape = 'ovo',kernel='rbf',verbose = True))
svc = SVC(decision_function_shape = 'ovo',kernel='rbf',verbose = True)

svc.fit(X_train, y_train)


logger.info('training done start predicting')

type(svc)
score =  svc.score(X_test, y_test)
logger.info(score)

