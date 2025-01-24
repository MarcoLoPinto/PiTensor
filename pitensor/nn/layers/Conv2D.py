import numpy as np
from .Layer import Layer

class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # Initialize weights and biases
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(1. / in_channels)
        self.biases = np.random.randn(out_channels, 1) * np.sqrt(1. / out_channels)
        self.input = None

    def forward(self, input):
        self.input = input
        batch_size, in_channels, height, width = input.shape
        out_height = height - self.kernel_size + 1
        out_width = width - self.kernel_size + 1
        
        # Prepare output array
        output = np.zeros((batch_size, self.out_channels, out_height, out_width))
        
        # Convolution operation using efficient broadcasting
        for i in range(out_height):
            for j in range(out_width):
                # Extract the window for convolution
                region = input[:, :, i:i+self.kernel_size, j:j+self.kernel_size]
                # region.shape is (batch_size, in_channels, kernel_size, kernel_size)
                region = region.reshape(batch_size, -1)  # Flatten the regions
                # weights.shape needs to be (out_channels, in_channels*kernel_size*kernel_size)
                weights_flat = self.weights.reshape(self.out_channels, -1)
                # Efficient matrix multiplication
                output[:, :, i, j] = np.dot(region, weights_flat.T) + self.biases.T
        
        return output
    
    def backward(self, grad_output):
        """
        Backpropagate through the convolution layer.
        grad_output has shape (batch_size, out_channels, out_height, out_width)
        """
        batch_size, in_channels, height, width = self.input.shape
        out_height = grad_output.shape[2]
        out_width = grad_output.shape[3]
        _, _, kernel_height, kernel_width = self.weights.shape
        
        # Initialize gradients
        grad_input = np.zeros_like(self.input)
        grad_weights = np.zeros_like(self.weights)
        grad_biases = np.sum(grad_output, axis=(0, 2, 3), keepdims=True).reshape(self.out_channels, 1)

        # Iterate over the output height and width
        for i in range(out_height):
            for j in range(out_width):
                # Extract regions from input corresponding to the current gradient output
                input_region = self.input[:, :, i:i+kernel_height, j:j+kernel_width]
                input_region_reshaped = input_region.reshape(batch_size, -1)
                
                # Reshape grad_output appropriately for matrix multiplication
                grad_out_current = grad_output[:, :, i, j].reshape(batch_size, self.out_channels)
                
                # Compute gradients for weights
                grad_weights += np.tensordot(grad_out_current.T, input_region_reshaped, axes=([1], [0])).reshape(self.out_channels, in_channels, kernel_height, kernel_width)
                
                # Compute gradient with respect to the input
                grad_out_expanded = grad_out_current.dot(self.weights.reshape(self.out_channels, -1))
                grad_input[:, :, i:i+kernel_height, j:j+kernel_width] += grad_out_expanded.reshape(batch_size, in_channels, kernel_height, kernel_width)

        return grad_input
    
    def get_parameters(self):
        return {
            'type': self.__class__.__name__,
            'weights': self.weights,
            'biases': self.biases
        }
    
    def update_parameters(self, params):
        self.weights = params['weights']
        self.biases = params['biases']
