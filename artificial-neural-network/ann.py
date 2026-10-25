# Project 3: Artificial Neural Network
# Author: Jawad Ahmad Khan
# Purpose: Learn the XOR logical pattern using a neural network.

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Training inputs
X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

# Expected outputs
y = [0, 1, 1, 0]

# Create the neural network
model = MLPClassifier(
    hidden_layer_sizes=(4,),
    activation="tanh",
    solver="lbfgs",
    max_iter=5000,
    random_state=42
)

# Train the network
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

# Display results
print("Artificial Neural Network: XOR Classification")
print("---------------------------------------------")

for inputs, prediction in zip(X, predictions):
    print(f"Input: {inputs} -> Predicted Output: {prediction}")

# Evaluate on the same four examples used for training
accuracy = accuracy_score(y, predictions)
print(f"\nTraining accuracy: {accuracy * 100:.2f}%")
