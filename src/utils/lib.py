import logging

import numpy as np


def roundfloat(x, decimals=3):
    """
    Round a float to certain decimals
    """
    return round(x, decimals) if type(x) == float else x


def print_line():
    """
    Print a line
    """
    logging.info("-" * 50)


def printdict(d):
    """
    Print a dictionary
    """
    for k, v in d.items():
        logging.info(f"{k}: {v}")


def rmse(y_true, y_pred):
    """
    Calculate the RMSE
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mape(y_true, y_pred):
    """
    Calculate the MAPE
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
