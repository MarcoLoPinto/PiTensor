import numpy as np
from .Layer import Layer

import numpy as np
from .Layer import Layer

class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size):
        """
        2D Convolutional Layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the square kernel.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Xavier Initialization for stability
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(1. / in_channels)
        self.biases = np.zeros((out_channels, 1, 1)) # Shape: (out_channels, 1, 1) for broadcasting

    def forward(self, input):
        """
        Performs the forward pass (convolution).

        Args:
            input (np.ndarray): Shape (batch_size, in_channels, height, width)

        Returns:
            np.ndarray: Output feature map of shape (batch_size, out_channels, out_height, out_width)
        """
        self.input = input  # Save for backprop
        batch_size, in_channels, height, width = input.shape

        # Compute output dimensions
        out_height = height - self.kernel_size + 1
        out_width = width - self.kernel_size + 1

        # Initialize output feature maps
        output = np.zeros((batch_size, self.out_channels, out_height, out_width))

        # Perform convolution for each output channel
        for out_channel in range(self.out_channels):
            for in_channel in range(in_channels):
                # Convolve each input channel with its corresponding weight
                output[:, out_channel] += np.array([
                    self.correlate2d(input[b, in_channel], self.weights[out_channel, in_channel])
                    for b in range(batch_size)
                ])
            
            # Add bias (broadcasting over batch & spatial dimensions)
            output[:, out_channel] += self.biases[out_channel]

        return output

    def backward(self, grad_output) -> np.ndarray:
        """
        Computes the gradients for backpropagation.

        Args:
            grad_output (np.ndarray): Gradient of loss w.r.t output (batch_size, out_channels, out_height, out_width)

        Returns:
            np.ndarray: Gradient of the loss with respect to the input.
        """
        batch_size, in_channels, height, width = self.input.shape
        _, _, kernel_height, kernel_width = self.weights.shape

        # Initialize gradient arrays
        grad_input = np.zeros_like(self.input)
        grad_weights = np.zeros_like(self.weights)
        grad_biases = np.sum(grad_output, axis=(0, 2, 3), keepdims=True)

        # Compute gradients for weights and input
        for out_channel in range(self.out_channels):
            for in_channel in range(in_channels):
                # Compute weight gradients
                grad_weights[out_channel, in_channel] = np.sum([
                    self.correlate2d(self.input[b, in_channel], grad_output[b, out_channel])
                    for b in range(batch_size)
                ], axis=0)

                # Compute input gradients (full mode for proper backpropagation)
                grad_input[:, in_channel] += np.array([
                    self.correlate2d(grad_output[b, out_channel], np.flip(self.weights[out_channel, in_channel]))
                    for b in range(batch_size)
                ])
        
        self.grad_weights = grad_weights
        self.grad_biases = grad_biases

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

    def correlate2d(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Performs 2D correlation (valid mode) using only NumPy.

        The output size is:
        - out_weight = in_weight - kernel_weight + 1
        - out_height = in_height - kernel_height + 1
        
        Args:
            image (np.ndarray): Input 2D array (H, W).
            kernel (np.ndarray): Filter/kernel 2D array (kH, kW).
        
        Returns:
            np.ndarray: Output feature map after correlation.
        """
        H, W = image.shape
        kH, kW = kernel.shape
        outH, outW = H - kH + 1, W - kW + 1 # Output dimensions

        # Extract sliding window patches from the image
        shape = (outH, outW, kH, kW)
        strides = image.strides * 2  # Step size for each dimension
        image_patches = np.lib.stride_tricks.as_strided(image, shape=shape, strides=strides)

        # Perform element-wise multiplication and sum across kernel dimensions
        output = np.einsum('ijkl,kl->ij', image_patches, kernel)

        return output

