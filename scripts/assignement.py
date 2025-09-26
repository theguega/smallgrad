import pickle

import matplotlib.pyplot as plt
import numpy as np

from autodiff_numpy.nn import Linear, ReLU, Sequential
from autodiff_numpy.tensor import Tensor

# load data
DATA_PATH = "data/assignment-one-test-parameters.pkl"

with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)

w1 = Tensor(np.transpose(data["w1"]))
b1 = Tensor(data["b1"])
w2 = Tensor(np.transpose(data["w2"]))
b2 = Tensor(data["b2"])
w3 = Tensor(np.transpose(data["w3"]))
b3 = Tensor(data["b3"])
inputs = Tensor(data["inputs"])
targets_data = np.array(data["targets"]).reshape(-1, 1)
targets = Tensor(targets_data)

# build the model
model = Sequential(Linear(2, 10), ReLU(), Linear(10, 10), ReLU(), Linear(10, 1))

model.layers[0].weight = w1
model.layers[0].bias = b1
model.layers[2].weight = w2
model.layers[2].bias = b2
model.layers[4].weight = w3
model.layers[4].bias = b3

print("1. gradients of the first-layer weights and biases of the untrained network")

first_input = Tensor(inputs.data[0:1])
first_target = Tensor(targets.data[0:1])

y_hat = model(first_input)
loss = ((y_hat - first_target) ** 2) * 0.5
model.zero_grad()
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

# training loop
for epoch in range(epochs + 1):
    total_loss = 0
    for i in range(num_samples):
        current_input = Tensor(inputs.data[i : i + 1, :])
        current_target = Tensor(targets.data[i : i + 1, :])

        y_hat = model(current_input)
        loss = ((y_hat - current_target) ** 2) * 0.5
        total_loss += loss.data.item()

    mean_loss = total_loss / num_samples
    losses.append(mean_loss)
    print(f"Epoch {epoch}: Average Loss = {mean_loss:.4f}")

    if epoch == epochs:
        break

    model.zero_grad()
    # accumulate gradients from the entire dataset
    y_hat_tenosr = model(inputs)
    loss_tensor = ((y_hat_tenosr - targets) ** 2) * 0.5
    loss_tensor.sum()
    loss_tensor.backward()

    # update parameters with the average gradient
    for p in model.parameters():
        mean_gradient = p.gradient / num_samples
        p.data -= learning_rate * mean_gradient

EXPORT_GRAPH_PATH = "report/figures/assignment_1.png"
plt.figure(figsize=(10, 6))
plt.plot(range(epochs + 1), losses, marker="o")
plt.title("Training Curve: Average Loss vs. Epoch")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.xticks(range(epochs + 1))
plt.grid(True)
plt.savefig(EXPORT_GRAPH_PATH, dpi=300)
print("plot saved as", EXPORT_GRAPH_PATH)
plt.show()
