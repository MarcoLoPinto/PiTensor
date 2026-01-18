from typing import Tuple

from pitensor.models.SequentialModel import SequentialModel
from pitensor.nn.layers import Conv2D, Flatten, Linear, MaxPool2D, ReLU, Sequential


class LeNet(SequentialModel):
    """LeNet-style model. Uses ReLU activations instead of the original tanh and MaxPool2D instead of AveragePooling2D for better performance."""

    def __init__(
        self,
        input_size: int = 28,
        in_channels: int = 1,
        num_classes: int = 10,
        conv_channels: Tuple[int, int] = (6, 16),
        kernel_size: int = 5,
        fc_sizes: Tuple[int, int] = (120, 84),
    ):
        conv1_out = self._conv_out_size(input_size, kernel_size)
        pool1_out = self._pool_out_size(conv1_out, 2, 2)
        conv2_out = self._conv_out_size(pool1_out, kernel_size)
        pool2_out = self._pool_out_size(conv2_out, 2, 2)
        flattened_size = conv_channels[1] * pool2_out * pool2_out

        layers = Sequential(
            Conv2D(in_channels, conv_channels[0], kernel_size),
            ReLU(),
            MaxPool2D(pool_size=(2, 2)),
            Conv2D(conv_channels[0], conv_channels[1], kernel_size),
            ReLU(),
            MaxPool2D(pool_size=(2, 2)),
            Flatten(),
            Linear(flattened_size, fc_sizes[0]),
            ReLU(),
            Linear(fc_sizes[0], fc_sizes[1]),
            ReLU(),
            Linear(fc_sizes[1], num_classes),
        )

        super().__init__(layers=layers)

    @staticmethod
    def _conv_out_size(size: int, kernel_size: int, stride: int = 1, padding: int = 0) -> int:
        return (size + 2 * padding - kernel_size) // stride + 1

    @staticmethod
    def _pool_out_size(size: int, pool_size: int, stride: int) -> int:
        return (size - pool_size) // stride + 1
