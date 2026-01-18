from typing import Tuple, Union

import numpy as np
from .Layer import Layer

class Conv2D(Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        padding_mode: str = "zeros",
    ):
        """
        2D Convolutional Layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the square kernel.
            stride (int or tuple): Stride for the convolution.
            padding (int or tuple): Padding for height/width.
            padding_mode (str): Padding mode passed to np.pad ("zeros" is an alias for "constant").
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode

        # Xavier Initialization for stability
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(1. / in_channels)
        self.biases = np.zeros((out_channels, 1, 1)) # Shape: (out_channels, 1, 1) for broadcasting

    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass (convolution).

        Args:
            input (np.ndarray): Shape (batch_size, in_channels, height, width)

        Returns:
            np.ndarray: Output feature map of shape (batch_size, out_channels, out_height, out_width)
        """
        self.input = input  # Save for backprop
        batch_size, in_channels, height, width = input.shape
        stride_h, stride_w = self._normalize_pair(self.stride)
        pad_h, pad_w = self._normalize_pair(self.padding)

        if pad_h > 0 or pad_w > 0:
            input_padded = np.pad(
                input,
                ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                mode=self._normalize_padding_mode(self.padding_mode)
            )
        else:
            input_padded = input

        # Compute output dimensions
        padded_height = input_padded.shape[2]
        padded_width = input_padded.shape[3]
        out_height = (padded_height - self.kernel_size) // stride_h + 1
        out_width = (padded_width - self.kernel_size) // stride_w + 1

        # Initialize output feature maps
        output = np.zeros((batch_size, self.out_channels, out_height, out_width))

        # Perform convolution for each output channel
        for out_channel in range(self.out_channels):
            for in_channel in range(in_channels):
                # Convolve each input channel with its corresponding weight
                output[:, out_channel] += np.array([
                    self.correlate2d(
                        input_padded[b, in_channel],
                        self.weights[out_channel, in_channel],
                        stride=(stride_h, stride_w)
                    )
                    for b in range(batch_size)
                ])
            
            # Add bias (broadcasting over batch & spatial dimensions)
            output[:, out_channel] += self.biases[out_channel]

        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Computes gradients for backpropagation.

        Args:
            grad_output (np.ndarray): Gradient of loss w.r.t output (batch_size, out_channels, out_height, out_width)

        Returns:
            np.ndarray: Gradient of loss w.r.t input (same shape as input).
        """
        batch_size, in_channels, height, width = self.input.shape
        _, _, kernel_height, kernel_width = self.weights.shape
        stride_h, stride_w = self._normalize_pair(self.stride)
        pad_h, pad_w = self._normalize_pair(self.padding)

        if pad_h > 0 or pad_w > 0:
            input_padded = np.pad(
                self.input,
                ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                mode=self._normalize_padding_mode(self.padding_mode)
            )
        else:
            input_padded = self.input

        padded_height, padded_width = input_padded.shape[2], input_padded.shape[3]

        # Gradients are computed on the padded input, then unpadded.
        grad_input_padded = np.zeros((batch_size, in_channels, padded_height, padded_width))
        grad_weights = np.zeros_like(self.weights)
        grad_biases = np.sum(grad_output, axis=(0, 2, 3), keepdims=False).reshape(self.out_channels, 1, 1)

        out_height, out_width = grad_output.shape[2], grad_output.shape[3]

        for out_channel in range(self.out_channels):
            for in_channel in range(in_channels):
                # Accumulate grad_weights using strided positions.
                for b in range(batch_size):
                    for out_y in range(out_height):
                        in_y = out_y * stride_h
                        for out_x in range(out_width):
                            in_x = out_x * stride_w
                            region = input_padded[b, in_channel, in_y:in_y + kernel_height, in_x:in_x + kernel_width]
                            grad_weights[out_channel, in_channel] += region * grad_output[b, out_channel, out_y, out_x]

                            grad_input_padded[b, in_channel, in_y:in_y + kernel_height, in_x:in_x + kernel_width] += (
                                self.weights[out_channel, in_channel] * grad_output[b, out_channel, out_y, out_x]
                            )

        self.grad_weights = grad_weights
        self.grad_biases = grad_biases

        if pad_h > 0 or pad_w > 0:
            return grad_input_padded[:, :, pad_h:pad_h + height, pad_w:pad_w + width]

        return grad_input_padded


    def get_parameters(self):
        return {
            'type': self.__class__.__name__,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'padding_mode': self.padding_mode,
            'weights': self.weights,
            'biases': self.biases,
        }

    def update_parameters(self, params):
        self.in_channels = params['in_channels']
        self.out_channels = params['out_channels']
        self.kernel_size = params['kernel_size']
        self.stride = params.get('stride', 1)
        self.padding = params.get('padding', 0)
        self.padding_mode = params.get('padding_mode', "constant")
        self.weights = params['weights']
        self.biases = params['biases']

    def correlate2d(
        self,
        image: np.ndarray,
        kernel: np.ndarray,
        stride: Union[int, Tuple[int, int]] = (1, 1),
    ) -> np.ndarray:
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
        stride_h, stride_w = self._normalize_pair(stride)
        H, W = image.shape
        kH, kW = kernel.shape
        outH = (H - kH) // stride_h + 1
        outW = (W - kW) // stride_w + 1 # Output dimensions

        # Extract sliding window patches from the image
        shape = (outH, outW, kH, kW)
        strides = (image.strides[0] * stride_h, image.strides[1] * stride_w, image.strides[0], image.strides[1])
        image_patches = np.lib.stride_tricks.as_strided(image, shape=shape, strides=strides)

        # Perform element-wise multiplication and sum across kernel dimensions
        output = np.einsum('ijkl,kl->ij', image_patches, kernel)

        return output

    def _normalize_pair(self, value: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
        if isinstance(value, (tuple, list)):
            return int(value[0]), int(value[1])
        return int(value), int(value)

    def _normalize_padding_mode(self, padding_mode: str) -> str:
        if padding_mode == "zeros":
            return "constant"
        return padding_mode
