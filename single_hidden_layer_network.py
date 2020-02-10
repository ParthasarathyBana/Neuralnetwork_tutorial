"""
Parthasarathy Bana
Personal Robotics Group


To complete the following assignment, I referred to these two websites;

https://towardsdatascience.com/building-an-artificial-neural-network-using-pure-numpy-3fe21acc5815

https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795

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
import matplotlib.pyplot as plt


# This is a class for a LinearTransform layer which takes an input 
# weight matrix W and computes W x as the forward step
class LinearTransform(object):
    # In this function you instantiate the member variables
    def __init__(self, W, b):
        self.weight_matrix = W
        self.bias_matrix = b
        self.weight_momentum_matrix = np.zeros((self.weight_matrix.shape))
        self.bias_momentum_matrix = np.zeros((self.bias_matrix.shape))

    # This is a fucntion for calculating the feed forward linear transform
    def forward(self, x):
        self.input_matrix = x
        z_output = np.matmul(self.input_matrix, self.weight_matrix) + self.bias_matrix
        return z_output

    # This is a function for calculating the backward propagation for linear transform
    def backward(self, grad_output, momentum = 0.1, learning_rate=1e-5):
        y = grad_output
        grad_input = np.matmul(y, np.transpose(self.weight_matrix))
        grad_weights = np.matmul(np.transpose(self.input_matrix), y) #np.transpose(np.dot(np.transpose(y), self.input_matrix))  
        grad_biases = np.sum(y, axis = 0)

        self.weight_momentum_matrix = momentum * self.weight_momentum_matrix - learning_rate * grad_weights
        self.weight_matrix += self.weight_momentum_matrix
        self.bias_momentum_matrix = momentum * self.bias_momentum_matrix - learning_rate * grad_biases
        self.bias_matrix += self.bias_momentum_matrix

        return grad_input, grad_weights, grad_biases

# This is a class for a ReLU layer max(x,0)
class ReLU(object):
    # This is a function to calculate the Relu activation for feed forward transform
    def forward(self, x):
        self.x = x
        self.relu_linear = np.maximum(0, self.x)
        return self.relu_linear

    # This is a function to calculate the Relu activation for backward propagation transform
    def backward(self, grad_output):
        self.relu_gradient = grad_output
        self.relu_gradient[self.relu_linear < 0.0] = 0.0
        return self.relu_gradient

# This is a class for a sigmoid layer followed by a cross entropy layer, the reason 
# this is put into a single layer is because it has a simple gradient form
class SigmoidCrossEntropy(object):
    # This is a function to calculate the Sigmoid activation for feed forward transform
    def forward(self, x):
        self.sigmoid_output = (1 / (1 + np.exp(-x)))
        return self.sigmoid_output

    # This is a function to calculate the Sigmoid activation for backward propagation transform 
    def backward(self, y, t):
        self.sigmoid_grad = y - t
        return self.sigmoid_grad

# This is a class for the Multilayer perceptron
class MLP(object):

    def __init__(self, input_dims, hidden_units):
    # INSERT CODE for initializing the network
        self.w1 = np.random.randn(input_dims, hidden_units) * 0.01
        self.b1 = np.zeros((1, hidden_units))
        self.w2 = np.random.randn(hidden_units, 1) * 0.01
        self.b2 = np.zeros((1, 1))

        self.linear_transform1 = LinearTransform(self.w1, self.b1)
        self.relu = ReLU()
        self.linear_transform2 = LinearTransform(self.w2, self.b2)
        self.sigmoid_entropy = SigmoidCrossEntropy()    

    def loss_function(self, pred_y, target_y, l2_penalty=0.001):
        loss = -1 *( target_y * np.log(pred_y + 10e-8) + (1 - target_y) * np.log(1 - pred_y + 10e-8))
        loss += l2_penalty / 2 * (np.sum(np.square(self.w1)) + np.sum(np.square(self.w2)))
        return loss

    def train(self, x_batch, y_batch, l2_penalty=0.001):
        linear_z1_in = self.linear_transform1.forward(x_batch)
        linear_z1_out = self.relu.forward(linear_z1_in)
        linear_z2_in = self.linear_transform2.forward(linear_z1_out)
        linear_z2_out = self.sigmoid_entropy.forward(linear_z2_in)
        loss = self.loss_function(linear_z2_out, y_batch, l2_penalty)

        linear_z2_backward = self.sigmoid_entropy.backward(linear_z2_out, y_batch)
        d_x2, d_w2, d_b2 = self.linear_transform2.backward(linear_z2_backward)
        linear_z1_backward = self.relu.backward(d_x2)
        d_x1, d_w1, d_b1 = self.linear_transform1.backward(linear_z1_backward)

        average_loss = np.mean(loss)
        average_accuracy = 100 * np.mean((np.round(linear_z2_out) == y_batch))
        return average_accuracy, average_loss        

    def evaluate(self, x, y):
        z1_in = self.linear_transform1.forward(x)
        z1_out = self.relu.forward(z1_in)
        z2_in = self.linear_transform2.forward(z1_out)
        z2_out = self.sigmoid_entropy.forward(z2_in)
        pred_y = z2_out
        validation_loss = np.mean(self.loss_function(pred_y, y, l2_penalty = 0.01))
        average_val_loss = np.mean(validation_loss)
        average_val_acc = 100*np.mean((np.round(z2_out) == y))
        return average_val_loss, average_val_acc

if __name__ == '__main__':
    if sys.version_info[0] < 3:
        data = pickle.load(open('cifar_2class_py2.p', 'rb'))
    else:
        data = pickle.load(open('cifar_2class_py2.p', 'rb'), encoding='bytes')

    train_x = data[b'train_data'] / 255.0
    train_y = data[b'train_labels']
    test_x = data[b'test_data']/255.0
    test_y = data[b'test_labels']

    num_examples, input_dims = train_x.shape
    num_epochs = 100
    batch_size = 256
    train_accuracy = []
    test_accuracy = []
    
    mlp = MLP(input_dims, hidden_units = 32)
    num_batches = num_examples // batch_size
    for epoch in range(num_epochs):
        rand_idx = np.arange(num_examples)
        np.random.shuffle(rand_idx)
        train_x = train_x[rand_idx]
        train_y = train_y[rand_idx]
        train_loss = 0.0
        train_acc = 0.0
        for batch in range(num_batches):
            x_batch = train_x[batch * batch_size: (batch + 1) * batch_size]
            y_batch = train_y[batch * batch_size: (batch + 1) * batch_size]
            accuracy, loss = mlp.train(x_batch, y_batch)
            train_loss += loss
            train_acc += accuracy
        train_loss = train_loss/num_batches
        train_acc = train_acc/num_batches
        train_accuracy.append(train_acc)  
        validation_loss, validation_acc = mlp.evaluate(test_x, test_y)
        test_accuracy.append(validation_acc)
        print(' Epoch:{}     Train Loss: {:.3f}    Train Acc.: {:.2f}%    Test Loss:  {:.3f}    Test Acc.:  {:.2f}%'.format(epoch, train_loss, train_acc, validation_loss, validation_acc))
    plt.plot(train_accuracy, label="train")
    plt.plot(test_accuracy, label="test")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.title("performance with learning_rate = 1e-5")
    plt.legend(loc="upper left")
    plt.show()
