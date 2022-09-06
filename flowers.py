from sklearn.preprocessing import scale
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import network as wee
from network import Network
import time

EPOCHS = 1000
LEARNING_RATE = 0.1
TRAINING_PORTION = 0.8


def split_data(data):
    inputs = []
    targets = []

    for row in trainingData:
        inputs.append(row[:-1])

        class_ = row[-1]
        if class_ == "Iris-setosa":
            targets.append([1, 0, 0])
        elif class_ == "Iris-versicolor":
            targets.append([0, 1, 0])
        elif class_ == "Iris-virginica":
            targets.append([0, 0, 1])
    
    return scale(np.array(inputs)), np.array(targets)


print("Preparing dataset")

# load data
# Structure:
# 0: sepal length
# 1: sepal width
# 2: petal length
# 3: petal width
# 4: class (0: setosa, 1: versicolor, 2: virginica)
data = pd.read_csv('datasets/iris.csv')

# convert to numpy array
data = data.to_numpy()

# shuffle data
np.random.shuffle(data)

print("Splitting into train/test")
# split into training and test data
trainingData = data[:int(len(data) * TRAINING_PORTION)]
testData = data[int(len(data) * TRAINING_PORTION):]

# split into inputs and targets
trainingInputs, trainingTargets = split_data(trainingData)
testInputs, testTargets = split_data(testData)

# Train network
print("Training network")
network = Network((4, 8, 3))

costs = network.train(trainingInputs, trainingTargets, LEARNING_RATE, EPOCHS)

# Get the train and test accuracy
print("Calculating costs")
trainAccuracy = network.accuracy(trainingInputs, trainingTargets)
testAccuracy = network.accuracy(testInputs, testTargets)

if input("Save model [Y/N]: ") == "Y":
    network.save("models/flowers.npz")

# Output
print(f"""\n\nTraining complete
Epochs:             {EPOCHS}
Learning rate:      {LEARNING_RATE}
Training portion:   {TRAINING_PORTION}
Training accuracy:  {trainAccuracy * 100}%
Test accuracy:      {testAccuracy * 100}%""")

# Plot cost
plt.plot(np.arange(0, len(costs)), costs, label='cost')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Cost (MSE)')
plt.show()