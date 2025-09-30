# 📦 NumPy Autodifferentiation Package

[![Python Version](https://img.shields.io/badge/python-≥3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![NumPy](https://img.shields.io/badge/numpy-≥2.3.3-orange.svg)](https://numpy.org/)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](https://docs.pytest.org/)
[![Linter](https://img.shields.io/badge/style-ruff-black.svg)](https://docs.astral.sh/ruff/)
[![Build](https://img.shields.io/badge/build-uv-yellow.svg)](https://github.com/astral-sh/uv)

A lightweight **automatic differentiation (autodiff) engine** and **neural network library** built entirely from scratch using **NumPy**.
This package can construct and train **multi-layer fully connected networks** with **ReLU** and **Sigmoid** nonlinearities.

---

## 📂 Project Structure

The repository is organized into three parts: **source code**, **scripts**, and **tests**.

```
.
├── scripts/
│   └── assignment.py             # Main script to run the assignment
├── src/
│   └── autodiff_numpy/
│       ├── __init__.py
│       ├── tensor.py             # Core autodiff engine (Tensor class)
│       └── nn.py                 # Neural network layers (Linear, ReLU, Sequential)
└── tests/
    ├── test_exercise.py          # Verifies the manual backprop exercise of previous week
    ├── test_nn.py                # Unit tests for the nn module
    └── test_tensor.py            # Unit tests for the Tensor class
```

---

## ⚡ Getting Started

### 1. Installation

Install dependencies with [uv](https://github.com/astral-sh/uv):

```bash
uv venv
source venv/bin/activate
uv pip install -e .
```

### 2. Run the Assignment

From the project root:

```bash
python scripts/assignment.py
```

This will:

1. Print initial gradients to the console.
2. Train the network for **5 epochs**, logging the average loss at each step.
3. Save a training curve plot as `report/figures/assignment_1.png`.

### 3. Run Tests

Verify correctness with:

```bash
uv run pytest
```

---

## 🛠️ Design & Implementation

The package is built around two core components: the **`Tensor` class** for autodiff and the **`nn` module** for neural networks.

### 🔹 The `Tensor` Class (`src/autodiff_numpy/tensor.py`)

The `Tensor` is a wrapper around a NumPy array that builds a computation graph dynamically.

* **Core idea:** Every operation between Tensors (`+`, `*`, `@`, etc.) creates a new Tensor, recording the operation and its parent nodes. This forms a DAG (Directed Acyclic Graph).
* **Key attributes:**

  * `data`: underlying NumPy array.
  * `gradient`: gradient of the final loss w.r.t. this Tensor.
  * `_previous_node`: parent Tensors in the graph.
  * `_backward`: local gradient function for the operation.

**Backpropagation (`.backward()`):**

1. Perform a **topological sort** of the graph.
2. Apply the **chain rule** in reverse order, propagating gradients back to all dependencies.

This allows automatic differentiation across arbitrary composed operations.

---

### 🔹 The `nn` Module (`src/autodiff_numpy/nn.py`)

A PyTorch-inspired minimal neural network library.

* **`Module`:** Base class with `parameters()` and `zero_grad()`.
* **`Linear`:** Fully connected layer (`output = input @ W + b`).
* **`ReLU`:** Nonlinear activation, no learnable parameters.
* **`Sigmoid`:** Sigmoid activation, no learnable parameters.
* **`Sequential`:** Simple layer container for chaining modules.

This modular design separates **low-level autodiff** from **high-level network architecture**, keeping the code clean and extensible.

---

## 📊 Results

### 1. Initial Gradients

The script outputs gradients of the first-layer weights and biases before training. Example:

```
Gradients of the first-layer weights:
 [[-0.0210 -0.0184  0.      0.0178  0.      0.
  -0.0097  0.      0.      0.    ]
 [-0.0926 -0.0815  0.      0.0783  0.      0.
  -0.0430  0.      0.      0.    ]]

Gradients of the first-layer biases:
 [-0.2151 -0.1893  0.      0.1820  0.      0.
  -0.0998  0.      0.      0.    ]
```

### 2. Training Loss

Training for 5 epochs with learning rate `0.01`:

```
Epoch 0: Average Loss = 0.5380
Epoch 1: Average Loss = 0.5350
Epoch 2: Average Loss = 0.5320
Epoch 3: Average Loss = 0.5291
Epoch 4: Average Loss = 0.5263
Epoch 5: Average Loss = 0.5236
```

### 3. Training Curve

The script generates a loss curve:

![Training Curve](report/figures/training_curve.png)

---

## ✍️ Credits

This README was written and polished with the help of **ChatGPT**  
The unit tests were written with the help of **Google Gemini 2.5 Pro**  
Implementation inspired by **micrograd**

> OpenAI. (2025). *ChatGPT* [Large language model]. Retrieved from [https://chat.openai.com](https://chat.openai.com)

> Google. (2025). *Gemini 2.5 Pro* [Large language model]. Retrieved from [https://aistudio.google.com](https://aistudio.google.com)

> Karpathy, A. (2020). *micrograd* [Autodiff engine]. Retrieved from [https://github.com/karpathy/micrograd/tree/master](https://github.com/karpathy/micrograd/tree/master)
