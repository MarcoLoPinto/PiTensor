from typing import Any
import numpy as np

class Tensor(np.ndarray):
    """A subclass of numpy.ndarray that behaves as a Tensor."""

    def __new__(self, data: Any, dtype=np.float32):
        """Creates a new Tensor instance."""
        obj = np.asarray(data, dtype=dtype).view(self)
        # obj.requires_grad = requires_grad # Extra attribute
        # obj.grad = None # Gradient placeholder
        return obj
    
    def __array_finalize__(self, obj):
        """Ensures Tensor properties are maintained."""
        if obj is None:
            return
        # self.requires_grad = getattr(obj, "requires_grad", False)
        # self.grad = getattr(obj, "grad", None)

    def __array_wrap__(self, out_arr, context=None, return_scalar=False):
        """Ensures all operations return a Tensor."""
        if return_scalar: # If NumPy expects a scalar, return a Python scalar
            return out_arr.item()
        return np.asarray(out_arr).view(Tensor)
    
    def numpy(self):
        """Explicitly converts Tensor back to a NumPy array."""
        return np.asarray(self)

def tensor(array: Any, dtype=np.float32):
    return Tensor(array, dtype=dtype)
