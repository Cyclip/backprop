from cProfile import label
from network import Network
import numpy as np
import matplotlib.pyplot as plt

EPOCHS = 5000
LEARNING_RATE = 0.2

xor = Network((2, 3, 1))

inputs = np.array((
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1)
))

targets = np.array((
    (0,),
    (1,),
    (1,),
    (0,)
))

costs = xor.train(inputs, targets, LEARNING_RATE, EPOCHS)

plt.plot(np.arange(0, len(costs)), costs, label='cost')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Cost (MSE)')
plt.show()