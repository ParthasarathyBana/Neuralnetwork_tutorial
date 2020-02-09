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


# This is a class for a LinearTransform layer which takes an input 
# weight matrix W and computes W x as the forward step
class LinearTransform(object):
    # In this function you instantiate the member variables
    def __init__(self, W, b):
        self.w = W
        self.b = b

    # This is a fucntion for calculating the feed forward linear transform
    def forward(self, x):
        self.x = x
        # print("Linear Transform (forward):")
        # print("x:", x.shape)
        # print("w:", self.w.shape)
        # print("b:", self.b.shape)
        z_output = np.dot(self.x, self.w) + self.b
        return z_output

    # This is a function for calculating the backward propagation for linear transform
    def backward(self, grad_output, learning_rate=1e-3, momentum=0.0, l2_penalty=0.0):
        # print("Linear Transform (backward):")
        y = grad_output
        # print("y_shape: ", y.shape)
        # print("x_shape: ", self.x.shape)
        # print("y_transpose_shape: ", np.transpose(y))
        # print("x_transpose_shape: ", np.transpose(self.x))
        grad_input = np.dot(y, np.transpose(self.w))
        grad_weights = np.dot(np.transpose(self.x), y) #np.transpose(np.dot(np.transpose(y), self.x))
        grad_biases = np.sum(y, axis = 0)
        self.w = self.w - learning_rate * grad_weights
        self.b = self.b - learning_rate * grad_biases
        return grad_input

# This is a class for a ReLU layer max(x,0)
class ReLU(object):
    # This is a function to calculate the Relu activation for feed forward transform
    def forward(self, x):
        self.x = x
        # print("Relu forward:")
        # print("x:", x.shape)
        self.relu_linear = np.maximum(0, self.x)
        return self.relu_linear

    # This is a function to calculate the Relu activation for backward propagation transform
    def backward(self, grad_output, learning_rate=0.0, momentum=0.0, l2_penalty=0.0):
        self.relu_gradient = self.x * grad_output
        return self.relu_gradient

# This is a class for a sigmoid layer followed by a cross entropy layer, the reason 
# this is put into a single layer is because it has a simple gradient form
class SigmoidCrossEntropy(object):
    # This is a function to calculate the Sigmoid activation for feed forward transform
    def forward(self, x):
        self.sigmoid_output = (1 / (1 + np.exp(-x)))
        return self.sigmoid_output

    # This is a function to calculate the Sigmoid activation for backward propagation transform 
    def backward(self, grad_output, learning_rate=0.0, momentum=0.0, l2_penalty=0.0):
        self.sigmoid_grad = (self.sigmoid_output - grad_output)
        return self.sigmoid_grad

# This is a class for the Multilayer perceptron
class MLP(object):

    def __init__(self, input_dims, hidden_units):
    # INSERT CODE for initializing the network
        self.w1 = np.random.randn(input_dims, hidden_units) * 0.1
        self.b1 = np.zeros((1, hidden_units))
        self.w2 = np.random.randn(hidden_units, 1) * 0.1
        self.b2 = np.zeros((1, 1))
        self.w1_m = 0
        self.b1_m = 0
        self.w2_m = 0
        self.b2_m = 0

        self.linear_transform1 = LinearTransform(self.w1, self.b1)
        self.relu = ReLU()
        self.linear_transform2 = LinearTransform(self.w2, self.b2)
        self.sigmoid_entropy = SigmoidCrossEntropy()    

    def loss_function(self, pred_y, target_y, l2_penalty):
        # print("z_foward:",type(pred_y), "target_y:", type(target_y))
        loss = -target_y*np.log(pred_y+1e-10)
        loss = np.mean(loss)
        return loss
        # loss = - target_y * np.log(pred_y) - (1 - target_y) * np.log(1 - pred_y)
        # loss += l2_penalty / 2 * (np.sum(np.square(self.w1)) + np.sum(np.square(self.w2)))
        # return loss

    def train(self, x_batch, y_batch, learning_rate = 1e-3, momentum = 0.5, l2_penalty=0):
        z_forward = self.linear_transform1.forward(x_batch)
        z_forward = self.relu.forward(z_forward)
        z_forward = self.linear_transform2.forward(z_forward)
        z_forward = self.sigmoid_entropy.forward(z_forward)
        # print("z_forward type: ", type(z_forward))
        loss = self.loss_function(z_forward, y_batch, l2_penalty)

        z_backward = self.sigmoid_entropy.backward(y_batch)
        z_backward = self.linear_transform2.backward(z_backward)
        z_backward = self.relu.backward(z_backward)
        z_backward = self.linear_transform1.backward(z_backward)

        average_loss = np.mean(self.loss_function(z_forward, y_batch, l2_penalty))
        average_accuracy = 100 * np.mean((np.round(z_forward) == y_batch))
        print(z_forward[:10])
        return average_accuracy, loss        

    def evaluate(self, x, y):
        z_forward = self.linear_transform1.forward(x_batch)
        z_forward = self.relu.forward(z_forward)
        z_forward = self.linear_transform2.forward(z_forward)
        z_forward = self.sigmoid_entropy.forward(z_forward)

        pred_y = np.round(z_forward)
        loss = np.mean(self.loss_function(pred_y, y))

    # INSERT CODE for testing the network
    # ADD other operations and data entries in MLP if needed
        pass

if __name__ == '__main__':
    if sys.version_info[0] < 3:
        data = pickle.load(open('cifar_2class_py2.p', 'rb'))
    else:
        data = pickle.load(open('cifar_2class_py2.p', 'rb'), encoding='bytes')

    train_x = data[b'train_data'] / 255
    train_y = data[b'train_labels'] / 255
    test_x = data[b'test_data']
    test_y = data[b'test_labels']
    # print(train_x.shape)
    num_examples, input_dims = train_x.shape
    # print("num_examples:", num_examples)
    # print("input_dims:", input_dims)
    # print(train_y.shape)
    # INSERT YOUR CODE HERE
    # YOU CAN CHANGE num_epochs AND num_batches TO YOUR DESIRED VALUES
    num_epochs = 10
    batch_size = 32
    average_train_accuracy = []
    average_train_loss = []

    mlp = MLP(input_dims, hidden_units = 10)

    for epoch in range(num_epochs):
        # random shuffling of data
        # train_x = train_x[np.random.shuffle(np.arange(num_examples))]
        # train_y = train_y[np.random.shuffle(np.arange(num_examples))]
        # print("after shuffling")
        # print("x_train: ",train_x.shape)
        num_batches = num_examples // batch_size
        for batch in range(num_batches):
            train_accuracy, train_loss = mlp.train(train_x, train_y)
            
            average_train_accuracy.append(np.mean(train_accuracy))
            average_train_loss.append(np.mean(train_loss))


            # INSERT YOUR CODE FOR EACH MINI_BATCH HERE
            # MAKE SURE TO UPDATE total_loss
            print('\r[Epoch {}, mb {}]    Avg.Loss = {:.3f}    Avg.Accuracy = {:.3f}'.format(epoch + 1,batch + 1,np.mean(average_train_loss), np.mean(average_train_accuracy)))
            sys.stdout.flush()
        # INSERT YOUR CODE AFTER ALL MINI_BATCHES HERE
        # MAKE SURE TO COMPUTE train_loss, train_accuracy, test_loss, test_accuracy
        print()
        print('    Train Loss: {:.3f}    Train Acc.: {:.2f}%'.format(average_train_loss,100.0 * average_train_accuracy))
        # print('    Test Loss:  {:.3f}    Test Acc.:  {:.2f}%'.format(
        #     test_loss,
        #     100. * test_accuracy,
        # ))
