"""
Parthasarathy Bana
Personal Robotics Group
"""


from __future__ import division
from __future__ import print_function

import sys
try:
   import _pickle as pickle
except:
   import pickle

import numpy as np
import random

# This is a class for a LinearTransform layer which takes an input 
# weight matrix W and computes W x as the forward step
class LinearTransform(object):

    def __init__(self, W, b):
    # DEFINE __init function
        self.w = W
        self.b = b

    def forward(self, x):
    # DEFINE forward function
        self.x = x
        z_output = np.dot(self.x, self.w) + self.b
        return z_output

    def backward(self, grad_output, learning_rate=1e-4, momentum=0.0, l2_penalty=0.0):
    # DEFINE backward function
    # ADD other operations in LinearTransform if needed
        y = grad_output
        grad_input = np.dot(y, np.transpose(self.w))
        grad_weights = np.transpose(np.dot(np.transpose(y), self.x))
        grad_biases = np.sum(grad_input, axis = 0)
        self.w = self.w - learning_rate * grad_weights
        self.b = self.b - learning_rate * grad_biases
        return grad_input

# This is a class for a ReLU layer max(x,0)
class ReLU(object):
    def forward(self, x):
    # DEFINE forward function
        self.x = x
        self.relu_linear = np.max(0, self.x)
        return self.relu_linear

    def backward(self, grad_output, learning_rate=0.0, momentum=0.0, l2_penalty=0.0):
    # DEFINE backward function
    # ADD other operations in ReLU if needed
        self.relu_gradient = self.x * grad_output
        return self.relu_gradient

# This is a class for a sigmoid layer followed by a cross entropy layer, the reason 
# this is put into a single layer is because it has a simple gradient form
class SigmoidCrossEntropy(object):
    def forward(self, x):
        # DEFINE forward function
        self.sigmoid_output = (1 / (1 + np.exp(-x)))
        return self.sigmoid_output

    def backward(self, grad_output, learning_rate=0.0, momentum=0.0, l2_penalty=0.0):
        # DEFINE backward function
        # ADD other operations and data entries in SigmoidCrossEntropy if needed
        self.sigmoid_grad = (self.sigmoid_output - grad_output)
        return self.sigmoid_output

# This is a class for the Multilayer perceptron
class MLP(object):

    def __init__(self, input_dims, hidden_units):
    # INSERT CODE for initializing the network
        w1 = np.random.randn(input_dims, hidden_units) * 0.01
        b1 = np.zeros(hidden_units, 1)
        w2 = np.random.randn(hidden_units, 1) * 0.01
        b2 = np.zeros(1, 1)

        self.linear_transform1 = LinearTransform(w1, b1)
        self.relu = ReLU()
        self.linear_transform2 = LinearTransform(w2, b2)
        self.sigmoid_entropy = SigmoidCrossEntropy()    

    def loss_function(pred_y, target_y):
        loss = - target_y * np.log(pred_y) - (1 - target_y) * np.log(1 - pred_y)

    def train(self, x_batch, y_batch, learning_rate, momentum, l2_penalty):
    # INSERT CODE for training the network
        z_forward = self.linear_transform1.forward(x_batch)
        z_forward = self.relu.forward(z_forward)
        z_forward = self.linear_transform2.forward(z_forward)
        z_forward = self.sigmoid_entropy.forward(z_forward)

        loss == self.loss_function(z_forward, y_batch)

        z_backward = self.sigmoid_entropy.backward(y_batch)
        z_backward = self.linear_transform2.backward(z_backward)
        z_backward = self.relu.backward(z_backward)
        z_backward = self.linear_transform1.backward(z_backward)

        return loss        

    def evaluate(self, x, y):
        z_forward = self.linear_transform1.forward(x_batch)
        z_forward = self.relu.forward(z_forward)
        z_forward = self.linear_transform2.forward(z_forward)
        z_forward = self.sigmoid_entropy.forward(z_forward)

        pred_y = np.round(z_forward)
        

    # INSERT CODE for testing the network
    # ADD other operations and data entries in MLP if needed
        pass

if __name__ == '__main__':
    if sys.version_info[0] < 3:
        data = pickle.load(open('cifar_2class_py2.p', 'rb'))
    else:
        data = pickle.load(open('cifar_2class_py2.p', 'rb'), encoding='bytes')

    train_x = data['train_data']
    train_y = data['train_labels']
    test_x = data['test_data']
    test_y = data['test_labels']
    
    num_examples, input_dims = train_x.shape
    # INSERT YOUR CODE HERE
    # YOU CAN CHANGE num_epochs AND num_batches TO YOUR DESIRED VALUES
    num_epochs = 10
    num_batches = 1000
    mlp = MLP(input_dims, hidden_units)

    for epoch in xrange(num_epochs):

    # INSERT YOUR CODE FOR EACH EPOCH HERE

        for b in xrange(num_batches):
            total_loss = 0.0
            # INSERT YOUR CODE FOR EACH MINI_BATCH HERE
            # MAKE SURE TO UPDATE total_loss
            print(
                '\r[Epoch {}, mb {}]    Avg.Loss = {:.3f}'.format(
                    epoch + 1,
                    b + 1,
                    total_loss,
                ),
                end='',
            )
            sys.stdout.flush()
        # INSERT YOUR CODE AFTER ALL MINI_BATCHES HERE
        # MAKE SURE TO COMPUTE train_loss, train_accuracy, test_loss, test_accuracy
        print()
        print('    Train Loss: {:.3f}    Train Acc.: {:.2f}%'.format(
            train_loss,
            100. * train_accuracy,
        ))
        print('    Test Loss:  {:.3f}    Test Acc.:  {:.2f}%'.format(
            test_loss,
            100. * test_accuracy,
        ))
