#set heading(numbering: "I.")
#set text(font:"Charter")

#set align(horizon)
#align(center)[#text(size: 18pt, weight: "bold", "Deep Learning - Assignment 1")
#v(2em)
#text(size: 12pt)[
*Names:* Daniel Szelepcsenyi, Marie De Mey, Theo Guegan, Pranav Panday

*Title:* Deep Learning Assignment 1

*Course:* SYDE 577

*Professor:* Dr. Bryan Tripp
]
]

#pagebreak()

#set heading(numbering: "1.a)")
#set par(
  justify: true,
)
= DESIGN OVERVIEW

The goal of this assignment was to develop an autodifferentiation package based on NumPy. Inspired by MiniGrad, the package has two main modules: the tensor module and the neural network module @minigrad. The tensor module creates the individual nodes, which allow the neural network module to connect these nodes, creating the full network.

== Tensor Module

The tensor module, implemented in `tensor.py`, was inspired by the Tensor datatype in PyTorch. The `Tensor` class takes in a NumPy array as an argument and initializes attributes for the associated gradient, the child nodes that it was created by, and a backward function depending on the operation that created it. The `_backward` function defines how the gradient is passed back to child nodes of the tensor during backpropagation. Before backpropagation, all gradients are reset to zero to ensure proper accumulation of the gradients. Backpropagation is performed by creating a topological order of all the tensors and visiting them recursively in inverted order. This method allows the use of the chain rule efficiently by propagating the gradient through the network.

For example, if `c = a + b` and `d = ReLU(c)`, reversing the topology will result in an order of `d, c, b, a`.

The tensor class has functions to override the built-in operations for addition, multiplication, exponentiation, and matrix multiplication, to work with tensors and define the `_backward` function. The `_backward` function is overloaded for each operator to assign the gradient with respect to the parameter of the tensor and also to the child one.

The tensor class also defines two non-linear activation functions: ReLU and Sigmoid, and their derivatives.

Here is a sample of the ReLU gradient calculation:

```python
a_grad = c.gradient * (self.data > 0)  # compute derivative with respect to self (a)
self.gradient = self.gradient + a_grad  # accumulate the gradient
```

#grid(
columns: (1fr, 1fr),
[
To illustrate this class’s functionality, let's take the loss tensor as an example: $L = 1/2(hat(y) - y)^2$ with $hat(y) = (h^2_1w^3_1 + h^2_2w^3_2) + b^3_1$. In this example, $y$ and $hat(y)$ are declared as child nodes of the square tensor and are saved in the `_previous_nodes` attribute as a set. @fig1 illustrates the computational graph developed during forward propagation and backpropagation. As seen, every operation creates a new node. The "grad" column indicates the gradient value during the forward pass, and the column under it represents the "calculated gradient" (c.g.) or δ calculated during the backward pass by reversing the depicted connections. The green boxes in Fig. 1 indicate the resulting value the weight will be updated by, and the δ values computed by hand in class that are required for earlier nodes in the network are shown in red boxes.
],
[
#figure(image("fig1.png"), caption: [Computational graph of nodes]) <fig1>
]
)

== Neural Network Module

The neural network module, implemented in `nn.py`, is used to construct multi-layer fully connected networks using the Tensor class as building blocks. Inspired by PyTorch, it builds a neural network by stacking multiple linear modules in the sequential module.

The module defines a base class called `Module`, which defines the methods `zero_grad` and `parameters`. This base class is inherited by the rest of the classes defined in the module, including the `Linear`, `ReLU`, `Sigmoid`, and `Sequential` classes. Overloading the `__call__` function defines the behavior of the network during the forward pass by calling the object with the input as a parameter.

The `Linear` class takes the number of input and output features (`in_features`, `out_features`) and initializes a weight tensor of size `[in_features, out_features]` following the Kaiming initialization discussed in the lectures. The bias of the neurons is initialized as a tensor of size `[1, out_features]` and set to zero. The weights and biases are saved as parameters of the module. The `__call__` function for the Linear class implements the forward pass: (`output = X@W + B`).

The `ReLU` and `Sigmoid` classes override the `__call__` function to call the tensor module implementation for the ReLU (`x.relu()`) or sigmoid (`x.sigmoid()`) functions, respectively, during the forward pass in the network.

The `Sequential` module allows multiple layers to be stacked and takes in a tuple of desired layers. The `__call__` function loops through the tuple of objects and feeds each module with the output of the previous one.

== Assignment Task Script (Model Training)

The script begins by importing our package, datasets, and pickle data file to set the weights, biases, inputs, and target values. The model is built using a fully connected feed-forward neural network made with the Sequential module as follows:

```python
model = Sequential(
  Linear(2, 10),
  ReLU(),
  Linear(10, 10),
  ReLU(),
  Linear(10, 1)
)
```

The input layer accepts two features, and the output layer produces a single scalar prediction (`ŷ`). The forward pass is done by feeding the input to the model (`y_hat = model(first_input)`), and the loss, in tensor form, is then calculated using MSE:

```python
loss = ((y_hat - first_target) ** 2) * 0.5
```

Afterward, the model's gradients are set to zero (`model.zero_grad()`) before a backward pass is performed on the loss tensor (`loss.backward()`). This process is first performed for only the first input-target pair to calculate the first-layer gradients and is then repeated in a training loop with the parameters adjusted each epoch according to the gradients and learning rate:

```python
p.data -= learning_rate * mean_gradient
```

== Results

#figure(
image("fig2.png", width: 70%), caption: [First-layer weights and biases of the untrained network for the first (input, target) pair in the training dataset]
) <fig2>

#figure(
image("fig3.png", width: 70%), caption: [Training curve with average loss over the dataset before any updates (with initial parameters) and after each update]
) <fig3>

#figure(
image("fig4.png", width: 70%), caption: [Training curve over a higher number of epochs to verify that the average loss decreases as the epoch number increases, thus verifying that the model is learning correctly]
) <fig4>

#bibliography("references.yaml", style: "institute-of-electrical-and-electronics-engineers")
