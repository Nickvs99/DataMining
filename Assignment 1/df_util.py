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

        if 10**(-exp_treshold) < abs(value) < 10**exp_treshold:
            return f"{value:.2f}"
        else:
            return f"{value:.2e}"
    
    with pd.option_context('display.max_colwidth', None):
        df.to_latex(path, float_format=lambda value:float_formatter(value))
