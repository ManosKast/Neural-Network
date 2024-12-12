from activation_functions import *
from optimisation import *
from cost_functions import *

# Converts an array of class labels into one-hot-encoded vectors.
def one_hot_encode(labels) -> np.ndarray:
    assert labels is not None, "Labels cannot be None"
    assert len(labels) > 0, "Labels cannot be empty"
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)

    # Create a zero matrix of shape (num_samples, num_classes)
    # Set the index corresponding to the label to 1 for each sample
    num_classes = len(np.unique(labels))
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot


def get_activation_function(activation_function: str, hyperparameter: float = 0):
    match activation_function:
        case 'linear':
            return Linear()
        case 'binary_step':
            return BinaryStep()
        case 'ReLU':
            return ReLU()
        case 'PReLU':
            return PReLU() if hyperparameter == 0 else PReLU(hyperparameter)
        case 'ELU':
            return ELU() if hyperparameter == 0 else ELU(hyperparameter)
        case 'SLU':
            return SLU() if hyperparameter == 0 else SLU(hyperparameter)
        case 'softplus':
            return Softplus()
        case 'logistic':
            return Logistic() if hyperparameter == 0 else Logistic(hyperparameter)
        case 'tanh':
            return Tanh()
        case 'arctan':
            return Arctan()
        case 'softmax':
            return Softmax()
        case _:
            raise ValueError(f'Activation function {activation_function} not recognised')
        

def get_optimisation_function(name: str) -> OptimisationFunction:
    if name == 'adam':
        return Adam()
    elif name == 'gradient_descent':
        return GradientDescent()
    else:
        raise ValueError("Invalid optimisation function name")
    

def get_cost_function(cost_function: str):
    cost_function = cost_function.lower()
    # TODO: Add hinge loss error function
    match cost_function:
        case 'l1_error':
            return L1Error()
        case 'l2_error':
            return L2Error()        
        case 'cross_entropy':
            return CrossEntropyLoss()
        case 'mean_squared_error':
            return MeanSquaredError()
        case 'mean_absolute_error':
            return MeanAbsoluteError()
        case 'binary_cross_entropy':
            return BinaryCrossEntropy()
        case _:
            raise ValueError(f'Cost function {cost_function} not recognised')