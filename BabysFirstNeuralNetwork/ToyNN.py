'''
Basic Python tutorial neural network.

Based on "A Neural Network in 11 Lines of Python" by i am trask
https://iamtrask.github.io/2015/07/12/basic-python-network/
'''

import numpy as np

class ToyNN(object):
    '''
    Simple two-layer toy neural network
    '''
    
    def __init__(self, inputs=3, outputs=1):
        #Number of input and output neurons
        self.inputs = inputs
        self.outputs = outputs
        
        #Initalize synapse weights randomly with a mean of 0
        self.synapseWeights = 2 * np.random.random((inputs, outputs)) - 1

    # Sigmoid activation function
    def Activation(self, x):
        return 1 / (1 + np.exp(-x))

    # Derivative of the sigmoid activation function
    def ActivationPrime(self, x):
        return x * (1 - x)

    # Forward propogation of inputs to outputs
    def ForwardPropogation(self, input):
        return self.Activation(np.dot(input, self.synapseWeights));

    # Training function
    def TrainNN(self, features, targets, iterations=10000):

        l0 = features #Input layer
        
        for iter in  range(iterations):
            
            #Forward propogation
            l1 = self.ForwardPropogation(l0) #output layer

            #Error calculation
            error = targets - l1

            #Back propogation
            # multiply slope by the error at each predicted value
            delta = error * self.ActivationPrime(l1)
            
            #update weights
            self.synapseWeights += np.dot(l0.T, delta)


# training features
features = np.array([   [0, 0, 1],
                        [0, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1] ])

# training targets
targets = np.array([ [0, 0, 1, 1] ]).T # 4x1 matrix

nn = ToyNN()
print("Training neural network...")
nn.TrainNN(features, targets)
print("Training complete.\n")

print("Input training set:")
print(targets)
print("Expected output:")
print(targets)
print("\nOutput from training set after 10000 iterations:")
print(nn.ForwardPropogation(features))

print("\n==============================\n")

newData = np.array([    [0, 0, 0],
                        [0, 1, 0],
                        [1, 0, 0] ])

print("New input data:")
print(newData)
print("Expected output:")
print(np.array([ [0, 0, 1] ]).T)
print("\nOutput for new data not in the training set:")
print(nn.ForwardPropogation(newData))