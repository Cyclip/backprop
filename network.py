import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Network:
    def __init__(self, shape, activation=sigmoid):
        """Initialises the network.
        Arguments:
            shape: A list of integers, where each integer represents the number of neurons in a layer.
            activation: The activation function to use. Defaults to sigmoid.
        
        Example:
            - net = Network([2, 3, 1])
            - net = Network([2, 3, 1], activation=relu)
        """
        self.shape = shape
        self.activation_function = activation
        self.neurons = np.fromiter([np.zeros(i) for i in shape], dtype=object)
    
        # Weights are stored in a 3D array.
        # The first dimension is the layer number.
        # The second dimension is the neuron number in the layer (of whose values is being feeded forward).
        # The third dimension is the neuron number in the next layer (to which the value is being feeded forward).
        self.weights = np.fromiter([np.random.randn(i, j) for i, j in zip(shape[:-1], shape[1:])], dtype=object)
        self.biases = np.fromiter([np.random.randn(i) for i in shape[1:]], dtype=object)
    
    def feed_forward(self, inputs):
        """
        Feeds forward a set of inputs through the network.
        Calculates the dot products of the inputs and the weights, and adds the biases.
        """
        self.neurons[0] = inputs
        for i in range(len(self.shape) - 1):
            self.neurons[i + 1] = self.activation_function(np.dot(self.neurons[i], self.weights[i]) + self.biases[i])
        
        return self.neurons[-1]


a = Network((2, 3, 1))