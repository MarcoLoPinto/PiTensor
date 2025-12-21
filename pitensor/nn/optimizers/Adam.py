from .Optimizer import Optimizer
import numpy as np

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, layers):
        self.t += 1
        for layer in layers:
            if not hasattr(layer, "weights") or not hasattr(layer, "grad_weights") or layer.grad_weights is None:
                continue

            layer_key = id(layer)
            if layer_key not in self.m:
                self.m[layer_key] = {'weights': np.zeros_like(layer.weights)}
                self.v[layer_key] = {'weights': np.zeros_like(layer.weights)}
                if hasattr(layer, "biases"):
                    self.m[layer_key]['biases'] = np.zeros_like(layer.biases)
                    self.v[layer_key]['biases'] = np.zeros_like(layer.biases)

            # Update biased first moment estimate
            self.m[layer_key]['weights'] = self.beta1 * self.m[layer_key]['weights'] + (1 - self.beta1) * layer.grad_weights

            # Update biased second raw moment estimate
            self.v[layer_key]['weights'] = self.beta2 * self.v[layer_key]['weights'] + (1 - self.beta2) * (layer.grad_weights ** 2)

            # Compute bias-corrected first moment estimate
            m_hat_weights = self.m[layer_key]['weights'] / (1 - self.beta1 ** self.t)

            # Compute bias-corrected second raw moment estimate
            v_hat_weights = self.v[layer_key]['weights'] / (1 - self.beta2 ** self.t)

            # Update weights
            layer.weights -= self.learning_rate * m_hat_weights / (np.sqrt(v_hat_weights) + self.epsilon)

            if hasattr(layer, "biases") and hasattr(layer, "grad_biases") and layer.grad_biases is not None:
                self.m[layer_key]['biases'] = self.beta1 * self.m[layer_key]['biases'] + (1 - self.beta1) * layer.grad_biases
                self.v[layer_key]['biases'] = self.beta2 * self.v[layer_key]['biases'] + (1 - self.beta2) * (layer.grad_biases ** 2)
                m_hat_biases = self.m[layer_key]['biases'] / (1 - self.beta1 ** self.t)
                v_hat_biases = self.v[layer_key]['biases'] / (1 - self.beta2 ** self.t)
                layer.biases -= self.learning_rate * m_hat_biases / (np.sqrt(v_hat_biases) + self.epsilon)
