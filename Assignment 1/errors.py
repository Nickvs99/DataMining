"""
A file with different error functions. 
"""


def MSE(actual, prediction):
    return (actual - prediction) ** 2
    pass

def MAE(actual, prediction):
    return abs(actual - prediction)
