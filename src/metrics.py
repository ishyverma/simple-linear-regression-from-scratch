import numpy as np

def _validate_inputs(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    if len(y_true) == 0:
        raise ValueError("Inputs cannot be empty")
    
    return y_true, y_pred

def mae(y_true, y_pred):
    """
    Mean Absolute Error
    """

    y_true, y_pred = _validate_inputs(y_true, y_pred)

    return np.mean(np.abs(y_true - y_pred))

def mse(y_true, y_pred):
    """
    Mean Squared Error
    """

    y_true, y_pred = _validate_inputs(y_true, y_pred)

    return np.mean((y_true - y_pred) ** 2)