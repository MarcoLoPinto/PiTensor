from pitensor.nn.layers import Softmax

import torch

import pandas as pd

from pitensor.Tensor import Tensor

Tensor.random.seed(42)

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

# Convert torch tensor to Tensor
x_tensor = Tensor(x_torch.detach().cpu().tolist())
grad_output_tensor = Tensor(grad_output_torch.detach().cpu().tolist())

# Forward pass in Tensor
s_tensor = softmax_numpy.forward(x_tensor)

# Backward pass in Tensor
grad_input_tensor = softmax_numpy.backward(grad_output_tensor)

# Display results
df_comparison = pd.DataFrame({
    "Torch Autograd": grad_autograd_torch.numpy().flatten(),
    "Tensor Manual": grad_input_tensor.flatten()
})

print(df_comparison)
