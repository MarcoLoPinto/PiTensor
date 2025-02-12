# Linear Layer

## Overview

The `Linear` class implements a **fully connected** (dense) layer in a neural network. This layer applies a **linear transformation** to the input:

```math
    Y = XW + B
```

where:

- $X$ is the input matrix of shape $(batch \_ size, input \_ dim)$
- $W$ is the weight matrix of shape $(input\_dim, output \_ dim)$
- $B$ is the bias vector of shape $(output \_ dim,)$
- $Y$ is the output matrix of shape $(batch \_ size, output \_ dim)$

The layer learns $W$ and $B$ during training through backpropagation, where:

- $\frac{\partial L}{\partial Y}$ has the same shape as $Y$

## Initialization

```python
self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(1. / input_dim)
self.biases = np.random.randn(output_dim) * np.sqrt(1. / output_dim)
```

Uses **Xavier (Glorot) Initialization** to maintain the scale of gradients across layers: the weights are initialized with a normal distribution scaled by $\sqrt{\frac{1}{input\_dim}}$.

In essence, Xavier initialization helps keep the magnitude of the outputs of the neurons similar across all layers and this helps in converging the model faster and prevents exploding and vanishing gradients.

## Forward

```python
self.input = input # Store input for backward pass
return np.dot(input, self.weights) + self.biases
```

- Computes the matrix multiplication $XW$
- Adds bias $B$
- Stores `input` for use in backpropagation

## Backward

### Gradient of loss w.r.t weights

```python
self.grad_weights = np.dot(self.input.T, grad_output)
```

To compute the gradient $\frac{\partial L}{\partial W}$ in order to be used to update the $W$ weights, we first use the chain rule:

$$
    \frac{\partial L}{\partial W} = 
    \frac{\partial L}{\partial Y} \cdot \frac{\partial Y}{\partial W}
$$

we assume that $\frac{\partial L}{\partial Y} = G$ has been computed (from the next layer of the forward pass or from a loss), so:

$$
    \frac{\partial L}{\partial W} = 
    G \cdot \frac{\partial Y}{\partial W} = 
    G \cdot \frac{\partial ( X \cdot W + B )}{\partial W} = X^T \cdot G
$$

where the resuling shape is: `(input_dim, output_dim)`. Maybe you are thinking: why the `grad_output` $G$ and $\frac{\partial Y}{\partial W}$ switched positions at the end of the computation ($X^T \cdot G$) ? Let's understand it better with an example (setting $B$ to zero to simplify computation):

$$
X  = \begin{bmatrix}
    x_{1,1} & x_{1,2} \\
    x_{2,1} & x_{2,2} \\
\end{bmatrix}
\quad
W  = \begin{bmatrix}
    w_{1,1} & w_{1,2} & w_{1,3} \\
    w_{2,1} & w_{2,2} & w_{2,3} \\
\end{bmatrix}
\quad
Y = \begin{bmatrix} 
    y_{1,1} & y_{1,2} & y_{1,3} \\ 
    y_{2,1} & y_{2,2} & y_{2,3} \\
\end{bmatrix}
$$

$$
Y = X \cdot W = \begin{bmatrix} 
    x_{1,1}w_{1,1} + x_{1,2}w_{2,1} & x_{1,1}w_{1,2} + x_{1,2}w_{2,2} & x_{1,1}w_{1,3} + x_{1,2}w_{2,3} \\ 
    x_{2,1}w_{1,1} + x_{2,2}w_{2,1} & x_{2,1}w_{1,2} + x_{2,2}w_{2,2} & x_{2,1}w_{1,3} + x_{2,2}w_{2,3} \\ 
\end{bmatrix}
$$

we get:

$$
\frac{\partial L}{\partial w_{1,1}} = 
\frac{\partial L}{\partial Y} \cdot \frac{\partial Y}{\partial w_{1,1}} =
\begin{bmatrix} 
    \frac{\partial L}{\partial y_{1,1}} & \frac{\partial L}{\partial y_{1,2}} & \frac{\partial L}{\partial y_{1,3}} \\ 
    \frac{\partial L}{\partial y_{2,1}} & \frac{\partial L}{\partial y_{2,2}} & \frac{\partial L}{\partial y_{2,3}} \\
\end{bmatrix}
\cdot
\begin{bmatrix}
    x_{1,1} & 0 & 0 \\
    x_{2,1} & 0 & 0 \\
\end{bmatrix} = 
\frac{\partial L}{\partial y_{1,1}} x_{1,1} + \frac{\partial L}{\partial y_{2,1}} x_{2,1}
$$

$$
\frac{\partial L}{\partial w_{1,2}} = 
... = 
\frac{\partial L}{\partial y_{1,2}} x_{1,1} + \frac{\partial L}{\partial y_{2,2}} x_{2,1}
$$

and so on, resulting in:

$$
\frac{\partial L}{\partial W} = \begin{bmatrix}
    x_{1,1} & x_{2,1} \\
    x_{1,2} & x_{2,2} \\
\end{bmatrix}
\begin{bmatrix} 
    \frac{\partial L}{\partial y_{1,1}} & \frac{\partial L}{\partial y_{1,2}} & \frac{\partial L}{\partial y_{1,3}} \\ 
    \frac{\partial L}{\partial y_{2,1}} & \frac{\partial L}{\partial y_{2,2}} & \frac{\partial L}{\partial y_{2,3}} \\
\end{bmatrix} = X^T \cdot \frac{\partial L}{\partial Y}
$$

### Gradient of loss w.r.t biases

```python
self.grad_biases = np.sum(grad_output, axis=0)
```

Computes the gradient $\frac{\partial L}{\partial B}$:

$$ 
    \frac{\partial L}{\partial B} = 
    \frac{\partial L}{\partial Y} \cdot \frac{\partial Y}{\partial B} = 
    \frac{\partial L}{\partial Y} \cdot \frac{\partial (XW + B)}{\partial B} = 
    \frac{\partial L}{\partial Y} \cdot I = 
    \sum_{i=1}^{batch\_size} \frac{\partial L}{\partial Y_i} =
    \sum_{rows} G
$$

with shape: `(output_dim,)`

### Gradient of loss w.r.t input

```python
grad_input = np.dot(grad_output, self.weights.T)
```

Computes the gradient of the loss with respect to the input $X$:

$$
    \frac{\partial L}{\partial X} = 
    \frac{\partial L}{\partial Y} \cdot \frac{\partial Y}{\partial X} = 
    \frac{\partial L}{\partial Y} \cdot \frac{\partial (XW + B)}{\partial X} = 
    \frac{\partial L}{\partial Y} \cdot W^T
$$

with shape `(batch_size, input_dim)`. Why here the $W^T$ does not swith positions? It is intuitive if we do the same procedure as in the example for the gradient loss w.r.t weights. 

As a general rule, remember (applies ONLY if a matrix A is independent of antoher one X): 

$$
    G \cdot \frac{\partial XW}{\partial W} = X^T \cdot G
$$

$$
    G \cdot \frac{\partial XW}{\partial X} = G \cdot W^T
$$

# TODO

## Overview

## Initialization

## Forward

## Backward

TODO
