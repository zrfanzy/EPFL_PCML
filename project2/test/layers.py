#!/usr/bin/env python
# -*- coding:utf8 -*-

import numpy
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

# RELU(x) = max(x, 0)
def relu(x):
    return T.switch(x<0, 0, x)

# ber error
def ber(pred, y):
  e = 0
  
  for i in range(0, 4):
    s = T.eq(y, i)
    er = T.neq(y, pred)
    [s, er] = [T.cast(x, dtype=theano.config.floatX) for x in [s, er]]
    e = e + T.sum(s * er) / T.sum(s)
  return e / numpy.float32(4)

# define logistic regression layer class
class LogisticRegression(object):

    # n_in: number of input; n_out: number of output
    def __init__(self, input, n_in, n_out, W=None, b=None):
        if W is None:
            W = theano.shared(
                              value=numpy.zeros( (n_in, n_out), dtype=theano.config.floatX),
                              name='W',
                              borrow=True)
        if b is None:
            b = theano.shared(
                              value=numpy.zeros( (n_out,), dtype=theano.config.floatX),
                              name='b',
                              borrow=True)
        self.W = W
        self.b = b
        
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean( T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
    
    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should hava the same shape as self.y_pred',
                            ('y', target.type, 'y_pred', self.y_pred,type))
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred,y))#ber(self.y_pred, y)#T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

# random number variables
rng = numpy.random.RandomState(12345)
srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))

# drop function: for dropout
def drop(input, p=0.95, rng=rng):
  mask = srng.binomial(n=1, p=p, size=input.shape, dtype=theano.config.floatX)
  return input * mask

# hidden layer class
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=None):
        self.input = input
        if W is None:
            W_values = numpy.asarray( rng.uniform(
                                                  low   = -numpy.sqrt(6. / (n_in + n_out)),
                                                  high  = numpy.sqrt(6. / (n_in + n_out)),
                                                  size  = (n_in, n_out)),
                                     dtype = theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values, name='W', borrow=True)

            if b is None:
                b_values = numpy.zeros( (n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
    
        self.W = W
        self.b = b
        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None else activation(lin_output))
        self.params = [self.W, self.b]

def _dropout_from_layer(rng, layer, p):
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    output = layer * T.cast(mask, theano.config.floatX)
    return output

class DropoutHiddenLayer(object):
  def __init__(self, rng, input, n_in, n_out, is_train=1, W=None, b=None,
      activation=None, p=0.95):
    self.input = input
    if W is None:
      W_values = numpy.asarray( rng.uniform(low = -numpy.sqrt(6. / (n_in +
        n_out)),high =
        numpy.sqrt(6./(n_in+n_out)),size=(n_in,n_out)),dtype=theano.config.floatX)
      if activation == theano.tensor.nnet.sigmoid:
        W_values *= 4
      W = theano.shared(value=W_values, name='W', borrow=True)
      if b is None:
        b_values = numpy.zeros( (n_out,),dtype=theano.config.floatX)
      b = theano.shared(value=b_values, name='b', borrow=True)

      self.W = W
      self.b = b
      lin_output = T.dot(input, drop(self.W)) + self.b
      output = lin_output
      if not(activation is None):
        output = activation(lin_output)

      train_output = drop(numpy.cast[theano.config.floatX](1./p)*output)
      self.output=T.switch(T.neq(is_train, 0), train_output, output)
      self.params= [self.W, self.b]

class MLP(object):
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        self.hiddenLayer = HiddenLayer( rng=rng, input=input,
                                       n_in=n_in, n_out=n_hidden,
                                       activation=None)
        self.logRegressionLayer = LogisticRegression(
                                                                                    input=self.hiddenLayer.output,
                                                                                    n_in=n_hidden,
                                                                                    n_out=n_out)
        self.L1     = abs(self.hiddenLayer.W).sum() + abs(self.logRegressionLayer.W).sum()
        self.L2_sqr = (self.hiddenLayer.W ** 2).sum() + (self.logRegressionLayer.W ** 2).sum()
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        self.errors = self.logRegressionLayer.errors
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

class LeNetConvLayer(object):
    def __init__(self, rng, input, filter_shape, image_shape, W=None, b=None,
        activation=None):
        assert image_shape[1] == filter_shape[1]
        self.input = input
        fan_in = numpy.prod(filter_shape[1:])
        fan_out = filter_shape[0] * numpy.prod(filter_shape[2:])
        
        if W is None:
            W_bound = numpy.sqrt(6. /(fan_in + fan_out))
            W = theano.shared(
                              numpy.asarray(
                                            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                                            dtype=theano.config.floatX),
                              borrow=True)
        if b is None:
            b_values = numpy.zeros( (filter_shape[0],), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, borrow=True)
        self.W = W
        self.b = b
        
        conv_out = conv.conv2d(
                               input = input,
                               filters = self.W,
                               filter_shape = filter_shape,
                               image_shape = image_shape)
        conv_out = (conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output = conv_out if activation is None else activation(conv_out)
        self.params = [self.W, self.b]

class PoolLayer(object):
    def __init__(self, input, poolsize=(2,2)):
        pooled_out = downsample.max_pool_2d(
                                            input = input,
                                            ds = poolsize,
                                            ignore_border = True)
        self.output = pooled_out

class LeNetConvPoolLayer(object):
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=None,
        W=None, b=None, activation=None):
        assert image_shape[1] == filter_shape[1]
        self.input = input
        fan_in = numpy.prod(filter_shape[1:])
        fan_out = filter_shape[0] * numpy.prod(filter_shape[2:])
        if not (poolsize is None):
          fan_out = fan_out / numpy.prod(poolsize)
        
        if W is None:
            W_bound = numpy.sqrt(6. /(fan_in + fan_out))
            W = theano.shared(
                              numpy.asarray(
                                            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                                            dtype=theano.config.floatX),
                              borrow=True)
        if b is None:
            b_values = numpy.zeros( (filter_shape[0],), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, borrow=True)
        self.W = W
        self.b = b
        
        conv_out = conv.conv2d(
                               input = input,
                               filters = self.W,
                               filter_shape = filter_shape,
                               image_shape = image_shape)
        conv_out = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')
        if not activation is None:
            conv_out = activation(conv_out)
            if poolsize is None:
                self.output = conv_out
            else:
                pooled_out = downsample.max_pool_2d(
                                                                   input = conv_out,
                                                                   ds = poolsize,
                                                                   ignore_border=True)
                self.output = pooled_out
            self.params = [self.W, self.b]
