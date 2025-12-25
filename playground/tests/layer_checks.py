import numpy as np

from pitensor.nn.layers import Conv2D, Flatten, Linear, MaxPool2D, ReLU, Softmax, Sequential


def numeric_grad(f, x, eps=1e-5):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        fx1 = f(x)
        x[idx] = orig - eps
        fx2 = f(x)
        x[idx] = orig
        grad[idx] = (fx1 - fx2) / (2 * eps)
        it.iternext()
    return grad


def rel_error(a, b, eps=1e-8):
    return np.max(np.abs(a - b) / (np.maximum(eps, np.abs(a) + np.abs(b))))


def run_check(name, fn):
    try:
        fn()
        print(f"PASS: {name}")
        return True
    except AssertionError as exc:
        print(f"FAIL: {name} - {exc}")
        return False


def check_linear():
    np.random.seed(0)
    layer = Linear(4, 3)
    x = np.random.randn(2, 4)
    upstream = np.random.randn(2, 3)

    out = layer.forward(x)
    assert out.shape == (2, 3), "forward shape mismatch"
    grad_input = layer.backward(upstream)
    assert grad_input.shape == x.shape, "backward shape mismatch"

    def f_input(x_in):
        return np.sum(layer.forward(x_in) * upstream)

    num_grad_input = numeric_grad(f_input, x.copy())
    err = rel_error(grad_input, num_grad_input)
    assert err < 1e-6, f"grad_input rel error too high: {err}"

    def f_weights(w):
        layer.weights = w
        return np.sum(layer.forward(x) * upstream)

    num_grad_weights = numeric_grad(f_weights, layer.weights.copy())
    err = rel_error(layer.grad_weights, num_grad_weights)
    assert err < 1e-6, f"grad_weights rel error too high: {err}"


def check_relu():
    np.random.seed(1)
    layer = ReLU()
    x = np.random.randn(3, 4) + 0.5
    out = layer.forward(x)
    assert np.all(out >= 0), "forward has negative values"
    grad_input = layer.backward(np.ones_like(out))
    assert np.all((x > 0) == (grad_input > 0)), "backward mask incorrect"


def check_softmax():
    np.random.seed(2)
    layer = Softmax()
    x = np.random.randn(2, 5)
    upstream = np.random.randn(2, 5)
    out = layer.forward(x)
    row_sums = np.sum(out, axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), "rows do not sum to 1"
    assert np.all(out > 0), "softmax outputs not strictly positive"

    grad_input = layer.backward(upstream)

    def f_input(x_in):
        return np.sum(layer.forward(x_in) * upstream)

    num_grad_input = numeric_grad(f_input, x.copy())
    err = rel_error(grad_input, num_grad_input)
    assert err < 1e-6, f"grad_input rel error too high: {err}"


def check_flatten():
    np.random.seed(3)
    layer = Flatten()
    x = np.random.randn(2, 3, 4, 5)
    out = layer.forward(x)
    assert out.shape == (2, 3 * 4 * 5), "forward shape mismatch"
    grad_out = np.random.randn(*out.shape)
    grad_input = layer.backward(grad_out)
    assert grad_input.shape == x.shape, "backward shape mismatch"


def check_conv2d():
    np.random.seed(4)
    layer = Conv2D(in_channels=1, out_channels=2, kernel_size=3)
    x = np.random.randn(2, 1, 5, 5)
    upstream = np.random.randn(2, 2, 3, 3)

    out = layer.forward(x)
    assert out.shape == upstream.shape, "forward shape mismatch"
    grad_input = layer.backward(upstream)
    assert grad_input.shape == x.shape, "backward shape mismatch"

    def f_input(x_in):
        return np.sum(layer.forward(x_in) * upstream)

    num_grad_input = numeric_grad(f_input, x.copy())
    err = rel_error(grad_input, num_grad_input)
    assert err < 1e-6, f"grad_input rel error too high: {err}"


def check_maxpool2d():
    np.random.seed(5)
    layer = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid")
    x = np.random.randn(2, 1, 4, 4)
    out = layer.forward(x)
    assert out.shape == (2, 1, 2, 2), "forward shape mismatch"
    upstream = np.random.randn(*out.shape)
    grad_input = layer.backward(upstream)
    assert grad_input.shape == x.shape, "backward shape mismatch"

    def f_input(x_in):
        return np.sum(layer.forward(x_in) * upstream)

    num_grad_input = numeric_grad(f_input, x.copy())
    err = rel_error(grad_input, num_grad_input)
    assert err < 1e-6, f"grad_input rel error too high: {err}"


def check_sequential():
    np.random.seed(6)
    layers = Sequential(Linear(4, 3), ReLU(), Linear(3, 2))
    x = np.random.randn(2, 4)
    out = layers.forward(x)
    assert out.shape == (2, 2), "forward shape mismatch"
    upstream = np.random.randn(2, 2)
    grad_input = layers.backward(upstream)
    assert grad_input.shape == x.shape, "backward shape mismatch"


def main():
    checks = [
        ("Linear", check_linear),
        ("ReLU", check_relu),
        ("Softmax", check_softmax),
        ("Flatten", check_flatten),
        ("Conv2D", check_conv2d),
        ("MaxPool2D", check_maxpool2d),
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
