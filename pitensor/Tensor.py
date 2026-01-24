from typing import Any
import numpy as np

class Tensor(np.ndarray):
    """A subclass of numpy.ndarray that behaves as a Tensor."""

    def __new__(self, data: Any, dtype=np.float64):
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

    def max(self, axis=None, out=None, keepdims=False, initial=None, where=True):
        """Max of array elements over a given axis."""
        return np.ndarray.max(self, axis=axis, out=out, keepdims=keepdims, initial=initial, where=where)

    def sum(self, axis=None, dtype=None, out=None, keepdims=False, initial=0, where=True):
        """Sum of array elements over a given axis."""
        return np.ndarray.sum(self, axis=axis, dtype=dtype, out=out, keepdims=keepdims, initial=initial, where=where)

    def argmax(self, axis=None, out=None):
        """Returns indices of the max values along an axis."""
        return np.ndarray.argmax(self, axis=axis, out=out)

    @staticmethod
    def array(data: Any, dtype=None) -> "Tensor":
        """Creates a Tensor from array-like data."""
        return Tensor(np.array(data, dtype=dtype))

    @staticmethod
    def asarray(data: Any, dtype=None) -> "Tensor":
        """Converts data to a Tensor."""
        return Tensor(np.asarray(data, dtype=dtype))

    @staticmethod
    def zeros(shape, dtype=np.float64) -> "Tensor":
        """Creates a Tensor filled with zeros."""
        return Tensor(np.zeros(shape, dtype=dtype))

    @staticmethod
    def zeros_like(a, dtype=None) -> "Tensor":
        """Creates a Tensor of zeros with the same shape as a."""
        return Tensor(np.zeros_like(a, dtype=dtype))

    @staticmethod
    def ones(shape, dtype=np.float64) -> "Tensor":
        """Creates a Tensor filled with ones."""
        return Tensor(np.ones(shape, dtype=dtype))

    @staticmethod
    def ones_like(a, dtype=None) -> "Tensor":
        """Creates a Tensor of ones with the same shape as a."""
        return Tensor(np.ones_like(a, dtype=dtype))

    @staticmethod
    def sqrt(x):
        """Elementwise square root."""
        return np.sqrt(x)

    @staticmethod
    def abs(x):
        """Elementwise absolute value."""
        return np.abs(x)

    @staticmethod
    def maximum(x1, x2):
        """Elementwise maximum."""
        return np.maximum(x1, x2)


    @staticmethod
    def exp(x):
        """Elementwise exponential."""
        return np.exp(x)

    @staticmethod
    def log(x):
        """Elementwise natural log."""
        return np.log(x)

    @staticmethod
    def concatenate(arrays, axis=0):
        """Concatenates a sequence of arrays."""
        return Tensor(np.concatenate(arrays, axis=axis))

    @staticmethod
    def pad(array, pad_width, mode="constant", **kwargs):
        """Pads an array and returns a Tensor."""
        return Tensor(np.pad(array, pad_width, mode=mode, **kwargs))

    @staticmethod
    def take_along_axis(arr, indices, axis):
        """Takes values along an axis using indices."""
        return np.take_along_axis(arr, indices, axis=axis)

    @staticmethod
    def einsum(subscripts, *operands, **kwargs):
        """Einstein summation convention."""
        return Tensor(np.einsum(subscripts, *operands, **kwargs))

    @staticmethod
    def as_strided(array, shape, strides):
        """Creates a strided view of the array."""
        return Tensor(np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides))

    @staticmethod
    def issubdtype(dtype, kind) -> bool:
        """Returns True if dtype is a subdtype of kind."""
        return np.issubdtype(dtype, kind)

    @staticmethod
    def iinfo(dtype):
        """Returns machine limits for integer types."""
        return np.iinfo(dtype)

    integer = np.integer
    int64 = np.int64
    float32 = np.float32
    float64 = np.float64
    inf = np.inf

    @staticmethod
    def save(file_path: str, data: Any, allow_pickle: bool = False) -> None:
        """Saves data to a .npy file."""
        np.save(file_path, np.asarray(data), allow_pickle=allow_pickle)

    @staticmethod
    def load(file_path: str, allow_pickle: bool = False) -> "Tensor":
        """Loads a tensor from a .npy file."""
        data = np.load(file_path, allow_pickle=allow_pickle)
        if getattr(data, "dtype", None) == object:
            return data
        return Tensor(data)


class _TensorRandom:
    def seed(self, seed=None) -> None:
        np.random.seed(seed)

    def randn(self, *shape, dtype=np.float64) -> Tensor:
        return Tensor(np.random.randn(*shape).astype(dtype))

    def rand(self, *shape, dtype=np.float64) -> Tensor:
        return Tensor(np.random.rand(*shape).astype(dtype))

    def randint(self, low, high=None, size=None, dtype=np.int64) -> Tensor:
        return Tensor(np.random.randint(low, high=high, size=size, dtype=dtype))

    def shuffle(self, x) -> None:
        np.random.shuffle(x)


Tensor.random = _TensorRandom()

def tensor(array: Any, dtype=np.float64):
    return Tensor(array, dtype=dtype)
