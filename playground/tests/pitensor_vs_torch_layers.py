from pitensor.nn.layers import Softmax

import torch

import numpy as np
import pandas as pd

np.random.seed(42)

# ---- PyTorch Implementation ----

# Create an example input
x_torch = torch.tensor([[2.0, 1.0, 0.1]], requires_grad=True)  # Example input

# Forward pass using PyTorch's built-in Softmax
softmax_layer_torch = torch.nn.Softmax(dim=1)
s_torch = softmax_layer_torch(x_torch)  # Forward pass

# Fake gradient coming from the loss function
grad_output_torch = torch.tensor([[1.0, 0.5, 0.2]])

# Compute backward using PyTorch autograd
s_torch.backward(grad_output_torch)
grad_autograd_torch = x_torch.grad.clone()

# ---- NumPy Implementation ----

# Create an instance of the NumPy-based softmax
softmax_numpy = Softmax()

# Convert torch tensor to numpy array
x_numpy = x_torch.detach().numpy()
grad_output_numpy = grad_output_torch.numpy()

# Forward pass in NumPy
s_numpy = softmax_numpy.forward(x_numpy)

# Backward pass in NumPy
grad_input_numpy = softmax_numpy.backward(grad_output_numpy)

# Display results
df_comparison = pd.DataFrame({
    "Torch Autograd": grad_autograd_torch.numpy().flatten(),
    "NumPy Manual": grad_input_numpy.flatten()
})

print(df_comparison)
