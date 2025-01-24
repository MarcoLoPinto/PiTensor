import numpy as np
from .Optimizer import Optimizer
from pitensor.nn.layers import Linear

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
        for i, layer in enumerate(layers):
            if isinstance(layer, Linear):
                if i not in self.m:
                    self.m[i] = {'weights': np.zeros_like(layer.weights), 'biases': np.zeros_like(layer.biases)}
                    self.v[i] = {'weights': np.zeros_like(layer.weights), 'biases': np.zeros_like(layer.biases)}
                
                # Update biased first moment estimate
                self.m[i]['weights'] = self.beta1 * self.m[i]['weights'] + (1 - self.beta1) * layer.grad_weights
                self.m[i]['biases'] = self.beta1 * self.m[i]['biases'] + (1 - self.beta1) * layer.grad_biases

                # Update biased second raw moment estimate
                self.v[i]['weights'] = self.beta2 * self.v[i]['weights'] + (1 - self.beta2) * (layer.grad_weights ** 2)
                self.v[i]['biases'] = self.beta2 * self.v[i]['biases'] + (1 - self.beta2) * (layer.grad_biases ** 2)

                # Compute bias-corrected first moment estimate
                m_hat_weights = self.m[i]['weights'] / (1 - self.beta1 ** self.t)
                m_hat_biases = self.m[i]['biases'] / (1 - self.beta1 ** self.t)

                # Compute bias-corrected second raw moment estimate
                v_hat_weights = self.v[i]['weights'] / (1 - self.beta2 ** self.t)
                v_hat_biases = self.v[i]['biases'] / (1 - self.beta2 ** self.t)

                # Update parameters
                layer.weights -= self.learning_rate * m_hat_weights / (np.sqrt(v_hat_weights) + self.epsilon)
                layer.biases -= self.learning_rate * m_hat_biases / (np.sqrt(v_hat_biases) + self.epsilon)
