import numpy as np

# Cost functions
def l1_error(y, y_pred):
    return np.sum(np.abs(y - y_pred))

def l2_error(y, y_pred):
    return np.sum(np.power(y - y_pred, 2))

def cross_entropy_loss(y, y_pred):
    return -np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

def mean_squared_error(y, y_pred):
    return np.mean(np.power(y - y_pred, 2))

def mean_absolute_error(y, y_pred):
    return np.mean(np.abs(y - y_pred))

def binary_crossentropy_error(y, y_pred):
    return np.mean(-y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred))

#TODO: Implement hinge loss error function
def hinge_loss_error(y, y_pred):
    pass

# Gradients of cost functions

# I did the maths on my own, hopefully it's correct. :D
def l1_error_gradient(y, y_pred):
    return np.sum((y_pred - y) / np.abs(y - y_pred))

def l2_error_gradient(y, y_pred):
    return np.sum(2 * (y_pred - y))

def cross_entropy_loss_gradient(y, y_pred):
    return np.sum(y / y_pred - (y - 1) / (y_pred - 1))

def mean_squared_error_gradient(y, y_pred):
    return np.mean(2 * (y_pred - y))

def mean_absolute_error_gradient(y, y_pred):
    return np.mean((y_pred - y) / np.abs(y - y_pred))

def binary_crossentropy_gradient(y, y_pred):
    return np.mean(y / y_pred - (y - 1) / (y_pred - 1))

def hinge_loss_gradient(y, y_pred):
    pass    