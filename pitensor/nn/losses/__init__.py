from .CrossEntropyLoss import CrossEntropyLoss

import sys, inspect
# Automatically fix redundant <class 'pitensor.nn.layers.Linear.Linear'> and so on
_current_module = sys.modules[__name__]  # Get current module dynamically
# Loop through all defined objects in this module
for name, obj in inspect.getmembers(_current_module):
    if inspect.isclass(obj) and obj.__module__.startswith(_current_module.__name__):
        obj.__module__ = _current_module.__name__  # Dynamically update the module name

__all__ = [
    "CrossEntropyLoss", 
]