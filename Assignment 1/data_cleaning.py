import numpy as np

def clean_df(df, method="remove"):
    """
    Clean the dataframe. Two approaches are possible:
        - remove: invalid values/outliers are set to NaN
        - median: invalid values/outliers are set to the median of the remaining data
    """

    df = df.copy()

    remove_impossible_values(df)
    remove_outliers(df)

    if method == "replace":
        replace_nan_values(df)

    return df


def remove_impossible_values(df):

    bounds_per_column = {
        "Students estimate": (0, 1000),
        "Stress level": (0, 100),
        "Sport": (0, 24 * 7),
    }

    for column in bounds_per_column:
        lower_bound, upper_bound = bounds_per_column[column]
        
        df[column] = np.where(
            (df[column] >= lower_bound) & (df[column] <= upper_bound) , df[column], np.NaN
        )

    # Remove any bedtime which does not follow the format hh:mm
    df["Bedtime"] = np.where(
        (df["Bedtime"].str.len() == 5) & (df["Bedtime"].str.get(2) == ":"), df["Bedtime"], np.NaN
    )


def remove_outliers(df, quantile=0.01):

    lower_quantile = quantile
    upper_quantile = 1 - quantile

    numerical_columns = df.select_dtypes(include=np.number)
    for column in numerical_columns:

        lower_bound = df[column].quantile(lower_quantile)
        upper_bound = df[column].quantile(upper_quantile)
        
        df[column] = np.where(
            (df[column] >= lower_bound) & (df[column] <= upper_bound) , df[column], np.NaN
        )

def replace_nan_values(df):

    numerical_columns = df.select_dtypes(include=np.number)
    for column in numerical_columns:

        value = df[column].median()

        df[column] = df[column].replace(np.nan, value)
