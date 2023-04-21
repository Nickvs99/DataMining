import numpy as np
import pandas as pd

def print_df(df):
    with pd.option_context('display.max_colwidth', None):
        print(df)

def print_header(header, char="="):

    print("\n")
    print(f"{char*10} {header} {char*(120 - len(header))}")
    print()

def save_df(df, path):

    def float_formatter(value, exp_treshold=3):
        """
        Print values with a low exponent using decimal notation.
        Print values with a large exponent using scientific notation.
        """

        if 10**(-exp_treshold) < abs(value) < 10**exp_treshold or value == 0:
            return f"{value:.2f}"
        else:
            return f"{value:.2e}"
    
    with pd.option_context('display.max_colwidth', None):
        df.to_latex(path, float_format=lambda value:float_formatter(value))

def normalize_df(df):
    """
    Normalizes the numerical columns. 
    Currently, the values are divided by the max value.
    Another possibility is to also shift the values such that the minvalue is 0
    """

    df = df.copy()

    numerical_columns = df.select_dtypes(include=np.number)
    for column in numerical_columns:

        max_value = df[column].max()

        df[column] /= max_value
    
    return df