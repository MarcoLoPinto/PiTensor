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
        self.pad_h = (0, 0)
        self.pad_w = (0, 0)
        if self.padding == 'same':
            pad_h = max((input.shape[2] - 1) // self.strides[0] * self.strides[0] + self.pool_size[0] - input.shape[2], 0)
            pad_w = max((input.shape[3] - 1) // self.strides[1] * self.strides[1] + self.pool_size[1] - input.shape[3], 0)
            self.pad_h = (pad_h // 2, pad_h - pad_h // 2)
            self.pad_w = (pad_w // 2, pad_w - pad_w // 2)
            pad_value = -np.inf
            if np.issubdtype(input.dtype, np.integer):
                pad_value = np.iinfo(input.dtype).min
            input = np.pad(
                input,
                ((0, 0), (0, 0), self.pad_h, self.pad_w),
                mode='constant',
                constant_values=(pad_value,)
            )
        return input

    def forward(self, input):
        """
        Forward pass for MaxPooling2D.

        Args:
            input (np.ndarray): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            np.ndarray: Pooled output of shape (batch_size, channels, out_height, out_width).
        """
        self.input_shape = input.shape
        input = self._pad_input(input)
        self.padded_shape = input.shape
        batch_size, channels, height, width = input.shape
        pool_height, pool_width = self.pool_size
        stride_height, stride_width = self.strides

        out_height = (height - pool_height) // stride_height + 1
        out_width = (width - pool_width) // stride_width + 1

        output = np.zeros((batch_size, channels, out_height, out_width))
        self.max_indices = np.zeros((batch_size, channels, out_height, out_width), dtype=np.int64)

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * stride_height
                h_end = h_start + pool_height
                w_start = j * stride_width
                w_end = w_start + pool_width

                window = input[:, :, h_start:h_end, w_start:w_end]
                window_reshaped = window.reshape(batch_size, channels, -1)
                max_indices = np.argmax(window_reshaped, axis=2)
                max_values = np.take_along_axis(window_reshaped, max_indices[:, :, None], axis=2)
                output[:, :, i, j] = max_values[:, :, 0]
                self.max_indices[:, :, i, j] = max_indices

        return output

    def backward(self, grad_output):
        """
        Backward pass for MaxPooling2D.

        Args:
            grad_output (np.ndarray): Gradient of loss w.r.t pooled output (batch_size, channels, out_height, out_width).

        Returns:
            np.ndarray: Gradient of loss w.r.t input (same shape as input).
        """
        grad_input = np.zeros(self.padded_shape, dtype=grad_output.dtype)
        out_height, out_width = grad_output.shape[2], grad_output.shape[3]
        pool_height, pool_width = self.pool_size
        stride_height, stride_width = self.strides

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * stride_height
                h_end = h_start + pool_height
                w_start = j * stride_width
                w_end = w_start + pool_width
                flat_index = self.max_indices[:, :, i, j]
                h_idx = flat_index // pool_width
                w_idx = flat_index % pool_width
                for b in range(grad_output.shape[0]):
                    for c in range(grad_output.shape[1]):
                        grad_input[b, c, h_start + h_idx[b, c], w_start + w_idx[b, c]] += grad_output[b, c, i, j]

        if self.padding == 'same':
            grad_input = grad_input[:, :, self.pad_h[0]:-self.pad_h[1] or None, self.pad_w[0]:-self.pad_w[1] or None]

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
