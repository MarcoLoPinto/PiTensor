from typing import Callable, List, Tuple
from itertools import product

import numpy as np

from pitensor.Tensor import Tensor

from pitensor.nn.layers import Conv2D, Flatten, Linear, MaxPool2D, ReLU, Softmax, Sequential


def numeric_grad(f: Callable[[Tensor], float], x: Tensor, eps: float = 1e-5) -> Tensor:
    """Compute numeric gradient of a scalar function using central differences."""
    grad = Tensor.zeros_like(x)
    for idx in product(*(range(dim) for dim in x.shape)):
        orig = x[idx]
        x[idx] = orig + eps
        fx1 = f(x)
        x[idx] = orig - eps
        fx2 = f(x)
        x[idx] = orig
        grad[idx] = (fx1 - fx2) / (2 * eps)
    return grad


def rel_error(a: Tensor, b: Tensor, eps: float = 1e-8) -> float:
    """Return the maximum relative error between two arrays."""
    diff = Tensor.abs(a - b)
    denom = Tensor.maximum(eps, Tensor.abs(a) + Tensor.abs(b))
    return (diff / denom).max()


def run_check(name: str, fn: Callable[[], None]) -> bool:
    """Run a single check, printing PASS/FAIL and returning success state."""
    try:
        fn()
        print(f"PASS: {name}")
        return True
    except AssertionError as exc:
        print(f"FAIL: {name} - {exc}")
        return False


def check_linear() -> None:
    """Validate Linear forward/backward shapes and gradients."""
    Tensor.random.seed(0)
    layer = Linear(4, 3)
    x = Tensor.random.randn(2, 4)
    upstream = Tensor.random.randn(2, 3)

    out = layer.forward(x)
    assert out.shape == (2, 3), "forward shape mismatch"
    grad_input = layer.backward(upstream)
    assert grad_input.shape == x.shape, "backward shape mismatch"

    def f_input(x_in: Tensor) -> float:
        return (layer.forward(x_in) * upstream).sum()

    num_grad_input = numeric_grad(f_input, x.copy())
    err = rel_error(grad_input, num_grad_input)
    assert err < 1e-6, f"grad_input rel error too high: {err}"

    def f_weights(w: Tensor) -> float:
        layer.weights = w
        return (layer.forward(x) * upstream).sum()

    num_grad_weights = numeric_grad(f_weights, layer.weights.copy())
    err = rel_error(layer.grad_weights, num_grad_weights)
    assert err < 1e-6, f"grad_weights rel error too high: {err}"


def check_relu() -> None:
    """Validate ReLU forward non-negativity and backward mask."""
    Tensor.random.seed(1)
    layer = ReLU()
    x = Tensor.random.randn(3, 4) + 0.5
    out = layer.forward(x)
    assert (out >= 0).all(), "forward has negative values"
    grad_input = layer.backward(Tensor.ones_like(out))
    assert ((x > 0) == (grad_input > 0)).all(), "backward mask incorrect"


def check_softmax() -> None:
    """Validate Softmax normalization and backward gradients."""
    Tensor.random.seed(2)
    layer = Softmax()
    x = Tensor.random.randn(2, 5)
    upstream = Tensor.random.randn(2, 5)
    out = layer.forward(x)
    row_sums = out.sum(axis=1)
    assert Tensor.abs(row_sums - 1.0).max() < 1e-6, "rows do not sum to 1"
    assert (out > 0).all(), "softmax outputs not strictly positive"

    grad_input = layer.backward(upstream)

    def f_input(x_in: Tensor) -> float:
        return (layer.forward(x_in) * upstream).sum()

    num_grad_input = numeric_grad(f_input, x.copy())
    err = rel_error(grad_input, num_grad_input)
    assert err < 1e-6, f"grad_input rel error too high: {err}"


def check_flatten() -> None:
    """Validate Flatten forward/backward shapes."""
    Tensor.random.seed(3)
    layer = Flatten()
    x = Tensor.random.randn(2, 3, 4, 5)
    out = layer.forward(x)
    assert out.shape == (2, 3 * 4 * 5), "forward shape mismatch"
    grad_out = Tensor.random.randn(*out.shape)
    grad_input = layer.backward(grad_out)
    assert grad_input.shape == x.shape, "backward shape mismatch"


def check_conv2d() -> None:
    """Validate Conv2D forward/backward shapes and input gradients."""
    Tensor.random.seed(4)
    layer = Conv2D(in_channels=1, out_channels=2, kernel_size=3)
    x = Tensor.random.randn(2, 1, 5, 5)
    upstream = Tensor.random.randn(2, 2, 3, 3)

    out = layer.forward(x)
    assert out.shape == upstream.shape, "forward shape mismatch"
    grad_input = layer.backward(upstream)
    assert grad_input.shape == x.shape, "backward shape mismatch"

    def f_input(x_in: Tensor) -> float:
        return (layer.forward(x_in) * upstream).sum()

    num_grad_input = numeric_grad(f_input, x.copy())
    err = rel_error(grad_input, num_grad_input)
    assert err < 1e-6, f"grad_input rel error too high: {err}"


def check_maxpool2d() -> None:
    """Validate MaxPool2D forward/backward shapes and input gradients."""
    Tensor.random.seed(5)
    layer = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid")
    x = Tensor.random.randn(2, 1, 4, 4)
    out = layer.forward(x)
    assert out.shape == (2, 1, 2, 2), "forward shape mismatch"
    upstream = Tensor.random.randn(*out.shape)
    grad_input = layer.backward(upstream)
    assert grad_input.shape == x.shape, "backward shape mismatch"

    def f_input(x_in: Tensor) -> float:
        return (layer.forward(x_in) * upstream).sum()

    num_grad_input = numeric_grad(f_input, x.copy())
    err = rel_error(grad_input, num_grad_input)
    assert err < 1e-6, f"grad_input rel error too high: {err}"


def check_maxpool2d_same_padding() -> None:
    """Validate MaxPool2D with padding='same' preserves input shape on backward."""
    Tensor.random.seed(7)
    layer = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")
    x = Tensor.random.randn(2, 1, 5, 5)
    out = layer.forward(x)
    upstream = Tensor.random.randn(*out.shape)
    grad_input = layer.backward(upstream)
    assert grad_input.shape == x.shape, "backward shape mismatch for padding='same'"


def check_maxpool2d_overlapping() -> None:
    """Validate MaxPool2D gradients with overlapping windows."""
    Tensor.random.seed(8)
    layer = MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding="valid")
    x = Tensor.random.randn(1, 1, 4, 4)
    out = layer.forward(x)
    upstream = Tensor.random.randn(*out.shape)
    grad_input = layer.backward(upstream)

    def f_input(x_in: Tensor) -> float:
        return (layer.forward(x_in) * upstream).sum()

    num_grad_input = numeric_grad(f_input, x.copy())
    err = rel_error(grad_input, num_grad_input)
    assert err < 1e-6, f"grad_input rel error too high: {err}"


def check_sequential() -> None:
    """Validate Sequential forward/backward shapes."""
    Tensor.random.seed(6)
    layers = Sequential(Linear(4, 3), ReLU(), Linear(3, 2))
    x = Tensor.random.randn(2, 4)
    out = layers.forward(x)
    assert out.shape == (2, 2), "forward shape mismatch"
    upstream = Tensor.random.randn(2, 2)
    grad_input = layers.backward(upstream)
    assert grad_input.shape == x.shape, "backward shape mismatch"

def check_tensor_numpy_equivalence() -> None:
    """Validate Tensor operations match NumPy for core ops."""
    np.random.seed(9)
    x_np = np.random.randn(3, 4)
    y_np = np.random.randn(4, 2)

    x = Tensor(x_np)
    y = Tensor(y_np)

    assert np.allclose(x @ y, x_np @ y_np), "matmul mismatch"
    assert np.allclose(x.sum(axis=1), x_np.sum(axis=1)), "sum mismatch"
    assert np.allclose(x.max(axis=0), x_np.max(axis=0)), "max mismatch"
    assert np.allclose(Tensor.exp(x), np.exp(x_np)), "exp mismatch"

    pos_np = np.abs(x_np) + 1e-3
    pos = Tensor(pos_np)
    assert np.allclose(Tensor.log(pos), np.log(pos_np)), "log mismatch"

    pad_np = np.pad(x_np, ((1, 1), (2, 2)), mode="constant")
    pad = Tensor.pad(x, ((1, 1), (2, 2)), mode="constant")
    assert np.allclose(pad, pad_np), "pad mismatch"

    assert np.allclose(x.argmax(axis=1), x_np.argmax(axis=1)), "argmax mismatch"
    idx_np = np.argmax(x_np, axis=1)
    idx = Tensor(idx_np, dtype=Tensor.int64)
    take_np = np.take_along_axis(x_np, idx_np[:, None], axis=1)
    take = Tensor.take_along_axis(x, idx[:, None], axis=1)
    assert np.allclose(take, take_np), "take_along_axis mismatch"

    a_np = np.random.randn(2, 3, 4)
    b_np = np.random.randn(4, 3)
    a = Tensor(a_np)
    b = Tensor(b_np)
    einsum_np = np.einsum("ijk,kl->ijl", a_np, b_np)
    einsum = Tensor.einsum("ijk,kl->ijl", a, b)
    assert np.allclose(einsum, einsum_np), "einsum mismatch"


def main() -> None:
    """Run all layer checks and exit non-zero on failure."""
    checks: List[Tuple[str, Callable[[], None]]] = [
        ("Tensor vs NumPy", check_tensor_numpy_equivalence),
        ("Linear", check_linear),
        ("ReLU", check_relu),
        ("Softmax", check_softmax),
        ("Flatten", check_flatten),
        ("Conv2D", check_conv2d),
        ("MaxPool2D", check_maxpool2d),
        ("MaxPool2D (same padding)", check_maxpool2d_same_padding),
        ("MaxPool2D (overlapping)", check_maxpool2d_overlapping),
        ("Sequential", check_sequential),
    ]
    passed = 0
    for name, fn in checks:
        if run_check(name, fn):
            passed += 1

    total = len(checks)
    print(f"{passed}/{total} checks passed.")
    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
