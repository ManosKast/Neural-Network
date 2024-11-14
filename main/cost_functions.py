import numpy as np

# Cost functions
def l1_error_function(y, y_pred):
    return np.sum(np.abs(y - y_pred))

def l2_error_function(y, y_pred):
    return np.sum(np.power(y - y_pred, 2))

def cross_entropy_loss_error_function(y, y_pred):
    return -np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

def mean_squared_error(y, y_pred):
    return np.mean(np.power(y - y_pred, 2))

def mean_absolute_error(y, y_pred):
    return np.mean(np.abs(y - y_pred))

def binary_crossentropy(y, y_pred):
    return np.mean(-y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred))

def hinge_loss_error_function(y, y_pred):
    return np.mean(np.maximum(0, 1 - y * y_pred))

# more to come