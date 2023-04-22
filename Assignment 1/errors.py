"""
A file with different error functions. 
"""


def MSE(actual, prediction):
    return (actual - prediction) ** 2

def MAE(actual, prediction):
    return abs(actual - prediction)
