from layers import *
from load_data import *

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

    def load_data(self, batch_size):
        datasets = load_data()
        self.train_set_x, self.train_set_y = datasets[0]
        self.valid_set_x, self.valid_set_y = datasets[1]
        #self.test_set_x, self.test_set_y = datasets[2]

        self.n_train_batches = self.train_set_x.get_value(borrow=True).shape[0] / batch_size
        self.n_valid_batches = self.valid_set_x.get_value(borrow=True).shape[0] / batch_size
        #self.n_test_batches = self.test_set_x.get_value(borrow=True).shape[0] / batch_size
        self.batch_size = batch_size

        print 'train_x: ', self.train_set_x.get_value(borrow=True).shape
        print 'train_y: ', self.train_set_y.shape.eval()
        print 'valid_x: ', self.valid_set_x.get_value(borrow=True).shape
        print 'valid_y: ', self.valid_set_y.shape.eval()
        #print 'test_x: ', self.test_set_x.shape
        #print 'test_y: ', self.test_set_y.shape

    def build_layer_architecture(self, acti_func=relu):
        self.index      = T.lscalar()
        self.step_rate  = T.dscalar()
        self.x = T.matrix('x')
        self.y = T.ivector('y')

        print 'building the model ...'
        
        layer6_input = self.x.flatten(2)

        self.layer6 = HiddenLayer(self.rng,
                input = layer6_input,
                n_in  = 36865,
                n_out = 4096,
                activation = acti_func)

        #self.layer7 = HiddenLayer(self.rng,
        #        input = self.layer6.output,
        #        n_in = 3072,
        #        n_out = 4096,
        #        activation = acti_func)

        self.layer8 = LogisticRegression(
                input = self.layer6.output,
                n_in = 4096,
                n_out = 4)

        self.cost = self.layer8.negative_log_likelihood(self.y)
    
    def build_test_valid_model(self):
        self.valid_model = theano.function(inputs=[self.index],
                outputs=self.layer8.errors(self.y),
                givens= {
                    self.x: self.valid_set_x[self.index * self.batch_size : (self.index+1) * self.batch_size],
                    self.y: self.valid_set_y[self.index * self.batch_size : (self.index+1) * self.batch_size]}
                )

    def build_test_test_model(self):
        self.test_model = theano.function(inputs=[self.index],outputs=self.softmax_layer.errors(self.y),
                givens={
                    self.x: self.test_set_x[self.index * self.batch_size : (self.index+1) * self.batch_size],
                    self.y: self.test_set_y[self.index * self.batch_size : (self.index+1) * self.batch_size]}
                 )

    def build_test_train_model(self):
        self.test_train_model = theano.function(inputs=[self.index],
              outputs = self.layer8.errors(self.y),
              givens = {
                self.x: self.train_set_x[self.index * self.batch_size :
                  (self.index + 1)* self.batch_size],
                self.y: self.train_set_y[self.index * self.batch_size :
                  (self.index + 1) * self.batch_size]}
        )


    def build_train_model(self):
        self.params = self.layer8.params + self.layer6.params 
        #        + self.layer6.params
        gparams = []
        for param in self.params:
            gparam = T.grad(self.cost, param)
            gparams.append(gparam)
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - self.step_rate * gparam))

        self.train_model = theano.function(inputs=[self.index, self.step_rate],
                outputs=self.cost,
                updates=updates,
                givens = {
                    self.x: self.train_set_x[self.index * self.batch_size : (self.index+1) * self.batch_size],
                    self.y: self.train_set_y[self.index * self.batch_size : (self.index+1) * self.batch_size]}
                )

    def train(self, n_epochs, learning_rate):
        print 'Training the model ...'
        train_sample_num = self.train_set_x.get_value(borrow=True).shape[0]
        #valid_sample_num = self.valid_set_x.get_value(borrow=True).shape[0]

        loss_records = []

        epoch = 0

        best_valid_loss = numpy.inf

        while epoch < n_epochs:
            train_losses = []
            for minibatch_index in xrange( self.n_train_batches ):
                minibatch_cost = self.train_model(minibatch_index, learning_rate)
                train_loss = self.test_train_model(minibatch_index)
                train_losses.append(train_loss)
                line = '\r\tepoch %i, minibatch_index %i/%i, error %f' % (epoch,
                    minibatch_index, self.n_train_batches, train_loss)
                sys.stdout.write(line)
                sys.stdout.flush()

            valid_losses = [self.valid_model(i) for i in xrange( self.n_valid_batches) ]
            #test_losses = [self.test_model(i) for i in xrange( self.n_test_batches)]

            train_score = numpy.mean(train_losses)
            valid_score = numpy.mean(valid_losses)

            #test_score = numpy.mean(test_losses)
            #loss_records.append((epoch, train_score, valid_score))
            #print '\nepoch %i, train_score %f, valid_score %f, test_score %f' % (epoch, train_score, valid_score, test_score)
            print '\nepoch %i, train_score %f, valid_score %f ' %(epoch,
                train_score, valid_score)

            params = [self.layer8.params, 
                      #self.layer7.params,
                      self.layer6.params, 
                      valid_score,
                      train_score,
                      epoch]

            if (valid_score < best_valid_loss):
                vest_valid_loss = valid_score
                with open('cnn.pkl', 'w') as f:
                    cPickle.dump(params, f)

            epoch += 1
        return loss_records


def make_training(learning_rate, n_epochs,
        batch_size, acti_func):
    cnn = CNN()
    cnn.load_data(batch_size)
    cnn.build_layer_architecture(acti_func)
    cnn.build_train_model()
    cnn.build_test_train_model()
    cnn.build_test_valid_model()
    cnn.train(n_epochs, learning_rate) 

if __name__ == '__main__':
    make_training(learning_rate=0.01, n_epochs=1000, batch_size=50, acti_func=relu)
