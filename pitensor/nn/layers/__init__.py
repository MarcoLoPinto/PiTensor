from .Layer import Layer

from .Linear import Linear
from .Conv2D import Conv2D

from .Sequential import Sequential
from .Flatten import Flatten

from .ReLU import ReLU
from .MaxPool2D import MaxPool2D

from .Softmax import Softmax

import sys, inspect
# Automatically fix redundant <class 'pitensor.nn.layers.Linear.Linear'> and so on
_current_module = sys.modules[__name__]  # Get current module dynamically
# Loop through all defined objects in this module
for name, obj in inspect.getmembers(_current_module):
    if inspect.isclass(obj) and obj.__module__.startswith(_current_module.__name__):
        obj.__module__ = _current_module.__name__  # Dynamically update the module name

__all__ = [
    "Layer", 
    "Linear",  "Conv2D", "MaxPool2D", "Flatten",
    "Sequential",
    "ReLU",
    "Softmax",
]