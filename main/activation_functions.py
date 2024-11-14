import numpy as np


# Cost functions

def linear_activation(x):
    return x

def binary_step_activation(x):
    return 1 if x >= 0 else 0

def ReLU_activation(x):
    return np.maximum(0, x)

# If x > 0, return x, else return alpha * x
# If x > 0, np.minimum(0, x) = 0, so the function returns x
# If x < 0, np.maximum(0, x) = 0, so the function returns alpha * x
def PReLU_activation(x, alpha=0.01):
    return np.maximum(0, x) + alpha * np.minimum(0, x)

def ELU_activation(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def SLU_activation(x, c=1.0):
    return x / (1 + np.exp(-c*x))

def softplus_activation(x):
    return np.log(1 + np.exp(x))


def logistic_activation(x, c=1.0):
    return 1 / (1 + np.exp(-c*x))

def tanh_activation(x):
    return np.tanh(x)

def arctan_activation(x):
    return np.arctan(x)


# Derivatives of cost functions

def linear_activation_derivative(x):
    return 1

def binary_step_activation_derivative(x):
    return 0

def ReLU_activation_derivative(x):
    return 1 if x >= 0 else 0

def PReLU_activation_derivative(x, alpha=0.01):
    return 1 if x >= 0 else alpha

def ELU_activation_derivative(x, alpha=1.0):
    return 1 if x >= 0 else alpha * np.exp(x)

def SLU_activation_derivative(x, c=1.0):
    return np.exp(-c*x) * (np.exp(c*x) + c*x + 1) / (np.exp(c*x) + 1)^2

def logistic_activation_derivative(x, c=1.0):
    return c * np.exp(-c*x) / (1 + np.exp(-c*x))^2

def softplus_activation_derivative(x):
    return 1 / (1 + np.exp(-x))

def tanh_activation_derivative(x):
    return 1 - np.tanh(x)^2

def arctan_activation_derivative(x):
    return 1 / (x^2 + 1)
