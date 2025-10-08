import pickle

import matplotlib.pyplot as plt
import numpy as np

from autodiff_numpy.nn import Linear, ReLU, Sequential
from autodiff_numpy.tensor import Tensor

# load data from pickle file containing pre-defined weights and test data
DATA_PATH = "data/assignment-one-test-parameters.pkl"

with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)

# extract weights and biases for each layer (transpose weights to match expected shape)
w1 = Tensor(np.transpose(data["w1"]))  # first layer weights: 2->10
b1 = Tensor(data["b1"])  # first layer biases
w2 = Tensor(np.transpose(data["w2"]))  # second layer weights: 10->10
b2 = Tensor(data["b2"])  # second layer biases
w3 = Tensor(np.transpose(data["w3"]))  # third layer weights: 10->1
b3 = Tensor(data["b3"])  # third layer biases

# prepare input data and targets
inputs = Tensor(data["inputs"])
targets_data = np.array(data["targets"]).reshape(-1, 1)
targets = Tensor(targets_data)

print(f"loaded data with {inputs.data.shape[0]} samples")
print(f"input shape: {inputs.data.shape}")
print(f"target shape: {targets.data.shape}")

# build the neural network: 2 inputs -> 10 hidden -> 10 hidden -> 1 output
# architecture: linear -> relu -> linear -> relu -> linear
model = Sequential(Linear(2, 10), ReLU(), Linear(10, 10), ReLU(), Linear(10, 1))

# assign the pre-loaded weights to the model layers
# note: layers[0], [2], [4] are linear layers (relu layers don't have weights)
model.layers[0].weight = w1  # first linear layer
model.layers[0].bias = b1
model.layers[2].weight = w2  # second linear layer
model.layers[2].bias = b2
model.layers[4].weight = w3  # third linear layer
model.layers[4].bias = b3

print("1. gradients of the first-layer weights and biases of the untrained network")

# take just the first sample to compute gradients
first_input = Tensor(inputs.data[0:1])
first_target = Tensor(targets.data[0:1])

print(f"using first sample: input = {first_input.data}, target = {first_target.data}")

# forward pass: compute prediction
y_hat = model(first_input)
print(f"model prediction: {y_hat.data}")

# compute loss using mean squared error formula: 0.5 * (y_hat - y)^2
loss = ((y_hat - first_target) ** 2) * 0.5
print(f"loss value: {loss.data}")

# reset gradients to zero before backward pass
model.zero_grad()
# backward pass: compute gradients
loss.backward()

print(
    "Gradients of the first-layer weights:\n",
    model.layers[0].weight.gradient,
)
print(
    "\nGradients of the first-layer biases :\n",
    model.layers[0].bias.gradient,
)

print()
print("---")
print()

print("2. train the network for five epochs with learning rate of 0.01")
learning_rate = 0.01
epochs = 5
num_samples = inputs.data.shape[0]
losses = []

print(f"starting training with {num_samples} samples")
print(f"learning rate: {learning_rate}")
print(f"number of epochs: {epochs}")

# training loop - each epoch processes all samples
for epoch in range(epochs + 1):
    # calculate loss (forward pass without gradients)
    total_loss = 0
    for i in range(num_samples):
        current_input = Tensor(inputs.data[i : i + 1, :])
        current_target = Tensor(targets.data[i : i + 1, :])

        y_hat = model(current_input)
        loss = ((y_hat - current_target) ** 2) * 0.5
        total_loss += loss.data.item()

    # compute and store average loss for this epoch
    mean_loss = total_loss / num_samples
    losses.append(mean_loss)

    if epoch == epochs:
        break

    # gradient computation and parameter update
    model.zero_grad()

    # forward pass on entire dataset to accumulate gradients
    y_hat_tensor = model(inputs)
    loss_tensor = ((y_hat_tensor - targets) ** 2) * 0.5
    total_loss_tensor = loss_tensor.sum()  # sum all losses
    total_loss_tensor.backward()  # compute gradients

    # update all model parameters using gradient descent
    # p.data = p.data - learning_rate * (gradient / num_samples)
    for p in model.parameters():
        mean_gradient = p.gradient / num_samples  # average gradient
        p.data -= learning_rate * mean_gradient

# create and save training loss plot
EXPORT_GRAPH_PATH = "scripts/assignment_1.png"
print("creating training curve plot...")

plt.figure(figsize=(10, 6))
plt.plot(range(epochs + 1), losses, marker="o", markersize=3)
plt.title("training curve: average loss vs. epoch")
plt.xlabel("epoch")
plt.ylabel("average loss")
plt.grid(True, alpha=0.3)

# save the plot
plt.savefig(EXPORT_GRAPH_PATH, dpi=300, bbox_inches="tight")
print(f"plot saved as {EXPORT_GRAPH_PATH}")

# show final training results
print("\ntraining completed!")
print(f"initial loss: {losses[0]:.6f}")
print(f"final loss: {losses[-1]:.6f}")
print(f"loss reduction: {((losses[0] - losses[-1]) / losses[0] * 100):.2f}%")

plt.show()
