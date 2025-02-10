import numpy as np

class MaxPool2D:
    """2D Max Pooling Layer."""

    def __init__(self, pool_size=(2, 2), strides=None, padding='valid'):
        """
        Initializes the MaxPool2D layer.

        Args:
            pool_size (tuple): Size of the pooling window (height, width).
            strides (tuple, optional): Strides of the pooling operation. Defaults to pool_size.
            padding (str): One of 'valid' or 'same'. 'valid' means no padding. 'same' results in padding
                           the input so that the output has the same length as the original input.
        """
        self.pool_size = pool_size
        self.strides = strides if strides is not None else pool_size
        self.padding = padding.lower()
        if self.padding not in {'valid', 'same'}:
            raise ValueError("Padding must be 'valid' or 'same'.")

    def _pad_input(self, input):
        if self.padding == 'same':
            pad_h = max((input.shape[2] - 1) // self.strides[0] * self.strides[0] + self.pool_size[0] - input.shape[2], 0)
            pad_w = max((input.shape[3] - 1) // self.strides[1] * self.strides[1] + self.pool_size[1] - input.shape[3], 0)
            pad_h = (pad_h // 2, pad_h - pad_h // 2)
            pad_w = (pad_w // 2, pad_w - pad_w // 2)
            input = np.pad(input, ((0, 0), (0, 0), pad_h, pad_w), mode='constant', constant_values=(0,))
        return input

    def forward(self, input):
        """
        Forward pass for MaxPooling2D.

        Args:
            input (np.ndarray): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            np.ndarray: Pooled output of shape (batch_size, channels, out_height, out_width).
        """
        input = self._pad_input(input)
        batch_size, channels, height, width = input.shape
        pool_height, pool_width = self.pool_size
        stride_height, stride_width = self.strides

        out_height = (height - pool_height) // stride_height + 1
        out_width = (width - pool_width) // stride_width + 1

        output = np.zeros((batch_size, channels, out_height, out_width))
        self.max_indices = np.zeros_like(input, dtype=bool)

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * stride_height
                h_end = h_start + pool_height
                w_start = j * stride_width
                w_end = w_start + pool_width

                window = input[:, :, h_start:h_end, w_start:w_end]
                max_values = np.max(window, axis=(2, 3), keepdims=True)
                output[:, :, i, j] = max_values.squeeze()
                self.max_indices[:, :, h_start:h_end, w_start:w_end] = (window == max_values)

        return output

    def backward(self, grad_output):
        """
        Backward pass for MaxPooling2D.

        Args:
            grad_output (np.ndarray): Gradient of loss w.r.t pooled output (batch_size, channels, out_height, out_width).

        Returns:
            np.ndarray: Gradient of loss w.r.t input (same shape as input).
        """
        grad_input = np.zeros_like(self.max_indices, dtype=grad_output.dtype)
        out_height, out_width = grad_output.shape[2], grad_output.shape[3]
        pool_height, pool_width = self.pool_size
        stride_height, stride_width = self.strides

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * stride_height
                h_end = h_start + pool_height
                w_start = j * stride_width
                w_end = w_start + pool_width

                grad_input[:, :, h_start:h_end, w_start:w_end] += (
                    self.max_indices[:, :, h_start:h_end, w_start:w_end] * grad_output[:, :, i, j][:, :, None, None]
                )

        if self.padding == 'same':
            pad_h = max((grad_input.shape[2] - 1) // stride_height * stride_height + pool_height - grad_input.shape[2], 0)
            pad_w = max((grad_input.shape[3] - 1) // stride_width * stride_width + pool_width - grad_input.shape[3], 0)
            pad_h = (pad_h // 2, pad_h - pad_h // 2)
            pad_w = (pad_w // 2, pad_w - pad_w // 2)
            grad_input = grad_input[:, :, pad_h[0]:-pad_h[1] or None, pad_w[0]:-pad_w[1] or None]

        return grad_input

    def get_parameters(self):
        return {
            'type': self.__class__.__name__,
            'pool_size': self.pool_size,
            'strides': self.strides,
            'padding': self.padding,
        }
    
    def update_parameters(self, params):
        self.pool_size = params['pool_size']
        self.strides = params['strides']
        self.padding = params['padding']