from layers import *
from load_test import *

import pickle
import cPickle
import gzip
import os
import sys
import time
import numpy
import theano

class CNN:
    def __init__(self):
        self.rng = numpy.random.RandomState(123456)

    def load_data(self):
        self.data = load_data()

    # build network
    def build_layer_architecture(self, w1, b1, w2, b2, acti_func=None):
        
        self.index      = T.lscalar()
        self.step_rate  = T.dscalar()
        self.x = T.matrix('x')
        self.y = T.ivector('y')

        print 'building the model ...'
        
        self.layer6_input = self.x.flatten(2)

        # fully connect layer: 36864 -> 512
        self.layer6_ = HiddenLayer(self.rng,
                input = self.layer6_input,
                W = w1,
                b = b1,
                n_in = 36865,
                n_out = 512,
                activation = acti_func)

        # classfication layer (using softmax afterwards) 
        self.layer8_ = LogisticRegression(
                input = self.layer6_.output,
                W = w2,
                b = b2,
                n_in = 512,
                n_out = 4)

    def build_predict_model(self):
        self.predict_model = theano.function(inputs=[self.layer6_input],
                outputs=self.layer8_.y_pred)

def make_predict(acti_func):
    # load network parameters
    if not (os.path.isfile('cnn3.pkl')):
        print 'download model file...'
        model = urllib.URLopener()
        model.retrieve('')

    f = open('cnn3.pkl', 'rb')
    params = pickle.load(f)
    f.close()
    w1 = params[1][0]
    b1 = params[1][1]
    w2 = params[0][0]
    b2 = params[0][1]
    cnn = CNN()
    cnn.load_data()
    cnn.build_layer_architecture(w1, b1, w2, b2, acti_func)
    
    cnn.build_predict_model()

    print 'getting prediction...'
    # get prediction
    pred = cnn.predict_model(cnn.data.get_value())
    # save
    a = {}
    a['pred'] = pred
    scipy.io.savemat('pred', a)

if __name__ == '__main__':
    make_predict(acti_func=None)
