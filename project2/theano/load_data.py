__docformat__ = 'restructedtext en'

import scipy.io
import cPickle
import gzip
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T


def load_data():

    mat_file_path = '/home/rzhou/XX_homework/EPFL_pcml/project2/train/train.mat'

    print 'load mat file'
    mat = scipy.io.loadmat(mat_file_path)

    all_data = mat['train']
    print 'mat file read finish, setting input and output'
    n = len(all_data['y'][0][0])

    train_set_y = numpy.arange(n, dtype = int)
    for i in range(0, n):
        train_set_y[i] = all_data['y'][0][0][i][0] - 1

    train_set_x = all_data['X_cnn'][0][0].reshape(n, 36865)
    train_set_x_hog = all_data['X_hog'][0][0].reshape(n, 5408)

    train_set = (train_set_x[0:4800], train_set_y[0:4800])
    valid_set = (train_set_x[4800:6000], train_set_y[4800:6000])
    train_set_hog = (train_set_x_hog[0:4800], train_set_y[0:4800])
    valid_set_hog = (train_set_x_hog[4800:6000], train_set_y[4800:6000])
    #test_set = (test_in, test_out)

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    #test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x_s, valid_set_y_s = shared_dataset(valid_set)
    train_set_x_s, train_set_y_s = shared_dataset(train_set)
    train_set_x_h, train_set_y_h = shared_dataset(train_set_hog)
    valid_set_x_h, valid_set_y_h = shared_dataset(valid_set_hog)

    rval = [(train_set_x_s, train_set_y_s), (valid_set_x_s, valid_set_y_s),
            (train_set_x_h, train_set_y_h), (valid_set_x_h, valid_set_y_h)]
    print 'data reading finished'
    #rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
    #        (test_set_x, test_set_y)]
    return rval

if __name__ == '__main__':
    dataset = load_data()
