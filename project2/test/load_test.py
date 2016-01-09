__docformat__ = 'restructedtext en'

import scipy.io
import cPickle
import gzip
import os
import sys
import timeit
import urllib

import numpy

import theano
import theano.tensor as T


def load_data():

    # check if test.mat is downloaded, if not, then download it
    if not(os.path.isfile('test.mat')):
        print 'downloading test.mat...'
        matfile = urllib.URLopener()
        matfile.retrieve('http://cvlabwww.epfl.ch/~cjbecker/tmp/test.mat', 'test.mat')
    
    mat_file_path = 'test.mat'

    print 'load mat file...'
    mat = scipy.io.loadmat(mat_file_path)

    all_data = mat['test']
    print 'mat file read finish, setting input...'

    n = len(all_data['X_cnn'][0][0])
    train_set_x = all_data['X_cnn'][0][0].reshape(n, 36865)
    print 'data load finished!'
    
    return theano.shared(numpy.asarray(train_set_x, dtype=theano.config.floatX),
            borrow=True)

if __name__ == '__main__':
    dataset = load_data()
