from cProfile import label
from network import Network
import numpy as np
import matplotlib.pyplot as plt

EPOCHS = 15000
LEARNING_RATE = 0.1

xor = Network((2, 3, 1), activation='sigmoid')

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
# xor.save("models/xor.npz")

print(f"Cost: {xor.cost(inputs, targets)}")

for inp, target in zip(inputs, targets):
    raw = xor.feed_forward(inp)[0]
    print(f"Input: {inp} | Output: {round(raw)} | Target: {target} | Raw: {raw}")

plt.plot(np.arange(0, len(costs)), costs, label='cost')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Cost (MSE)')
plt.show()