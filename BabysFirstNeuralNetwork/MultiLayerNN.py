import numpy as np

class MultiLayerNN(object):
    '''
    Expansion of the 2-layer toy NN to accomodate arbitrary hidden layers
    '''
    
    def __init__(self, inputs=3, outputs=1, hiddenLayers=[]):
        """ Initialize Neural Network
            
            inputs: number of network inputs
            outputs: number of network outputs
            hiddenLayers: list of ints > 0 where each item defines a hidden layer with n neurons
        """
        #Number of input and output neurons
        self.inputs = inputs
        self.outputs = outputs
        
        if len(hiddenLayers) > 0:
            inputSynapses = inputs 
            weights = []
            for layer in hiddenLayers:
                #Initalize synapse weights randomly with a mean of 0
                #The dimensionality of the synapse weights is defined by the layer's inputs and outputs
                weights.append(2 * np.random.random((inputSynapses, layer)) - 1)
                inputSynapses = layer #Input for the next layer is the output of this one. 
            weights.append(2 * np.random.random((inputSynapses, outputs)) - 1) #Output layer
            self.synapseWeights = np.array(weights)
        else:
            # No Hidden layers
            self.synapseWeights = np.array([2 * np.random.random((inputs, outputs)) - 1])

    # Sigmoid activation function
    def Activation(self, x):
        return 1 / (1 + np.exp(-x))

    # Derivative of the sigmoid activation function
    def ActivationPrime(self, x):
        return x * (1 - x)

    # Feed input into the trained network and receive output
    def Predict(self, input, layer=0):
        if (layer + 1 == len(self.synapseWeights)):
            return self.FeedForward(input, self.synapseWeights[layer])
        elif (layer + 1 < len(self.synapseWeights)):
            output = self.FeedForward(input, self.synapseWeights[layer])
            return self.Predict(output, layer = layer + 1)
        else:
            print("Error: layer index out of range.")
            return None

    # Forward propogation of inputs to outputs
    def FeedForward(self, input, weights):
        return self.Activation(np.dot(input, weights));

    # Training function
    def TrainNN(self, features, targets, iterations=10000, periods=10):

        l0 = features #Input layer
        
        for iter in  range(iterations):
            
            outputs = [l0] #Stores output values at each layer. Should probably be an np array for performance

            #Forward propogation
            for layer in range(len(self.synapseWeights)):
                outputs.append(self.FeedForward(outputs[layer], self.synapseWeights[layer]))

            ln = outputs[-1] #output layer

            #Error calculation
            error = targets - ln

            #Periodic output
            if (iter % (iterations / periods)) == 0:
                period = int(iter / (iterations / periods))
                print("[Period " + str(period) + "] Mean absolute error: " + str(np.mean(np.abs(error))))

            #Back propogation
            delta = error * self.ActivationPrime(ln)

            for layer in range(len(self.synapseWeights)-1, -1, -1):
                oldWeights = self.synapseWeights[layer]
                #update weights
                self.synapseWeights[layer] += np.dot(outputs[layer].T, delta)

                # calculate error and delta for previous layer
                error = delta.dot(oldWeights.T)
                delta = error * self.ActivationPrime(outputs[layer])


# training features
features = np.array([   [0, 0, 1],
                        [0, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1] ])

# training targets
targets = np.array([ [0, 1, 1, 0] ]).T # 4x1 matrix

# Seed random number generator
np.random.seed(1)

nn = MultiLayerNN(hiddenLayers=[4])
print("Training neural network...")
nn.TrainNN(features, targets, iterations = 60000)
print("Training complete.\n")

print("Input training set:")
print(features)
print("Expected output:")
print(targets)
print("\nOutput from training set after 60000 iterations:")
print(nn.Predict(features))

print("\n==============================\n")

newData = np.array([    [0, 0, 0],
                        [0, 1, 0],
                        [1, 0, 0] ])

print("New input data:")
print(newData)
print("\nOutput for new data not in the training set:")
print(nn.Predict(newData))