import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return 1. * (x > 0)

class Network:
    def __init__(self, shape, activation='sigmoid'):
        """Initialises the network.
        Arguments:
            shape: A list of integers, where each integer represents the number of neurons in a layer.
            activation: The activation function to use. Defaults to sigmoid.
        
        Example:
            - net = Network([2, 3, 1])
            - net = Network([2, 3, 1], activation=relu)
        """
        self.shape = shape
        self.neurons = [np.zeros(shape[i]) for i in range(len(shape))]
    
        # Weights are stored in a 3D array.
        # The first dimension is the layer number.
        # The second dimension is the neuron number in the layer (of whose values is being feeded forward).
        # The third dimension is the neuron number in the next layer (to which the value is being feeded forward).
        self.weights = np.fromiter([np.random.randn(i, j) for i, j in zip(shape[:-1], shape[1:])], dtype=object)
        self.biases = np.fromiter([np.random.randn(i) for i in shape[1:]], dtype=object)

        self.set_activation(activation)
    
    def set_activation(self, activation):
        if activation == 'sigmoid':
            self.activation_function = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation == 'relu':
            self.activation_function = relu
            self.activation_derivative = relu_derivative
        else:
            raise ValueError(f'Activation function {activation} not supported.')

    def feed_forward(self, inputs):
        """
        Feeds forward a set of inputs through the network.
        Calculates the dot products of the inputs and the weights, and adds the biases.
        """
        self.neurons[0] = inputs
        for i in range(len(self.shape) - 1):
            self.neurons[i + 1] = self.activation_function(np.dot(self.neurons[i], self.weights[i]) + self.biases[i])
        
        return self.neurons[-1]
    
    def cost(self, inputs, targets):
        """
        Calculates the cost of the network based on a single input.
        """
        outputs = np.array(self.feed_forward(inputs))
        return (1 / len(inputs)) * np.sum((targets - outputs) ** 2)

    def train(self, inputs, targets, learningRate=0.1, epochs=100):
        """
        Trains the network for n epochs.
        It first feeds forward the inputs, then backpropagates to reduce the cost.
        """
        costs = []

        for epoch in range(epochs):
            # if epoch % 100 == 0:
            #     print(f"[EPOCH {epoch + 1}/{epochs}] Starting epoch")
            for inp, target in zip(inputs, targets):
                self.backpropagate(inp, target, learningRate)
            
            cost = self.cost(inputs, targets)
            costs.append(cost)
        
        return costs
    
    def backpropagate(self, inputs, targets, learningRate):
        """
        Backpropagates the network to reduce the cost.
        Arguments:
            inputs: The inputs to feed forward.
            targets: The targets to compare the outputs to.
            learningRate: The learning rate to use.
        
        Example:
            - net.backpropagate([0, 0], [0], 0.1)
        """
        # Feed forward
        self.feed_forward(inputs)

        # Calculate error
        error = targets - self.neurons[-1]

        # Backpropagate
        for i in reversed(range(len(self.shape) - 1)):
            delta = error * self.activation_derivative(np.dot(self.neurons[i], self.weights[i]) + self.biases[i])
            self.weights[i] += learningRate * np.dot(self.neurons[i][:, None], delta[None, :])
            self.biases[i] += learningRate * delta
            error = np.dot(delta, self.weights[i].T)
    
    def save(self, path):
        """
        Saves the network to a file.
        """
        np.savez(path, shape=self.shape, weights=self.weights, biases=self.biases)
    
    @staticmethod
    def load(path):
        """
        Loads a network from a file.
        """
        data = np.load(path)
        net = Network(data['shape'])
        net.weights = data['weights']
        net.biases = data['biases']
        return net
    
    def predict(self, inputs):
        """
        Predicts the output of a set of inputs.
        """
        outputs = self.feed_forward(inputs)
        return outputs.argmax()