import matplotlib.pyplot as plt
import numpy as np

from df_util import print_df, print_header, save_df

def explore_df(df, **kwargs):
           
    basic_df = get_basic_df(df)
    numerical_df = get_numerical_df(df)
    categorical_df = get_categorical_df(df)

    show_basic_stats(basic_df, **kwargs)

    show_numerical_stats(df, numerical_df, **kwargs)
    show_categorical_stats(categorical_df, **kwargs)

    show_correlation_matrix(df, **kwargs)


def get_basic_df(df):

    def n_invalid(series):
        return series.size - series.count()

    def percentage_valid(series):
        return series.count() / series.size

    def most_frequent_value(series):
        return series.value_counts().idxmax()

    def most_frequent_count(series):
        return series.value_counts().max()

    return df.agg(["dtype", "size", n_invalid, "nunique"])


def get_numerical_df(df):

    def quantile_1(series):
        return series.quantile(0.01)

    def quantile_25(series):
        return series.quantile(0.25)
    
    def quantile_75(series):
        return series.quantile(0.75)
    
    def quantile_99(series):
        return series.quantile(0.99)

    numerical_columns = df.select_dtypes(include=np.number)
    
    return numerical_columns.agg(["mean", "std", "min", quantile_25, "median", quantile_75, "max"])


def get_categorical_df(df):

    def category_frequencies(series):
        
        value_counts = series.value_counts()

        categories = value_counts.index.tolist()
        frequencies = value_counts.values.tolist()

        # Combine categories with their respective frequency
        return list(zip(categories, frequencies))

    categorical_columns = df.select_dtypes(include='category')
    
    return categorical_columns.agg([category_frequencies])


def show_basic_stats(df, save_suffix=None, show=True, **kwargs):

    print_header("Basic stats")
    print_df(df.transpose())

    if save_suffix is not None:
        save_df(df.transpose(), f"tables/basic_stats{save_suffix}.tex")

    n_invalids = [df[column]["n_invalid"] for column in df.columns]

    labels = df.columns.values
    plt.barh(labels, n_invalids)
    plt.title("Number of invalid/missing entries")

    plt.tight_layout()

    if save_suffix is not None:
        plt.savefig(f"figures/invalids{save_suffix}.png", dpi=400)

    if show:
        plt.show()
    else:
        plt.close()



def show_numerical_stats(df, numerical_df, save_suffix=None, show=True):

    print_header("Numerical stats")
    print_df(numerical_df.transpose())
    save_df(numerical_df.transpose(), f"tables/numerical_stats{save_suffix}.tex")

    numerical_columns = df.select_dtypes(include=np.number)

    for column in numerical_columns:

        values = df[column].values
        
        plt.hist(values, bins=20)
        plt.axvline(numerical_df[column]["median"], label="median", color="orange", linestyle="--")

        plt.title(column)
        plt.ylabel("Frequency")
        
        plt.legend()   

        if save_suffix is not None:
            plt.savefig(f"figures/numeric_{column}{save_suffix}.png", dpi=400)

        if show:
            plt.show()
        else:
            plt.close()


def show_categorical_stats(df, save_suffix=None, show=True, **kwargs):

    print_header("Categorical stats")
    print_df(df.transpose())
    save_df(df.transpose(), f"tables/categorical_stats{save_suffix}.tex")
    
    for column in df:

        category_frequencies = df[column].category_frequencies

        categories = [category_frequency[0] for category_frequency in category_frequencies]
        frequencies = [category_frequency[1] for category_frequency in category_frequencies]

        plt.pie(frequencies, labels=categories)
        plt.title(column)

        if save_suffix is not None:
            plt.savefig(f"figures/category_{column}{save_suffix}.png", dpi=400)
        
        if show:
            plt.show()
        else:
            plt.close()

def show_correlation_matrix(df, save_suffix=None, show=True, **kwargs):
    
    df_copy = df.copy()

    # Convert all categorical data to numerical data
    # https://stackoverflow.com/a/32011969/12132063
    categorical_columns = df_copy.select_dtypes(['category']).columns
    df_copy[categorical_columns] = df_copy[categorical_columns].apply(lambda x: x.cat.codes)

    # Calculate the correlation matrix, for some reason method="pearson" does not give a matrix with one's on the diagonal
    correlation_matrix = df_copy.corr(method="kendall", numeric_only=True)

    print_header("Correlation matrix")
    print_df(correlation_matrix)

    plt.imshow(correlation_matrix)
    
    plt.title("Correlation matrix")

    labels = correlation_matrix.columns.values
    tick_locations = range(len(labels))
    plt.xticks(tick_locations, labels, rotation=30, horizontalalignment="right")
    plt.yticks(tick_locations, labels)

    plt.colorbar()
    plt.tight_layout()

    if save_suffix is not None:
        plt.savefig(f"figures/correlation{save_suffix}.png", dpi=400)

    if show:
        plt.show()
    else:
        plt.close()

def plot_scatterplot(df, index1, index2, save_suffix=None, show=True, **kwargs):

    values1 = df[index1].values
    values2 = df[index2].values
    
    plt.scatter(values1, values2)
    
    plt.xlabel(index1)
    plt.ylabel(index2)
    
    plt.tight_layout()
   
    if save_suffix is not None:
        plt.savefig(f"figures/scatter_{index1}_{index2}{save_filename}.png", dpi=400)

    if show:
        plt.show()
    else:
        plt.close()

def plot_boxplot(df, index1, indices, save_suffix=None, show=True, **kwargs):
    """
    Creates a boxplot for multiple columns.
    Index1 has to be a numerical column
    Indices has to be an array of categorical columns
    """

    data = []
    labels = []

    for index in indices:

        for category in df[index].cat.categories:

            labels.append(f"{index} - {category}")

            # Get all rows where the column value is equal to the current category
            rows = df.loc[df[index] == category]
            values = rows[index1].values
            
            # Remove nan values from the data
            values = values[~np.isnan(values)]

            data.append(values)

    plt.boxplot(data, labels=labels)

    plt.ylabel(index1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_suffix is not None:
        plt.savefig(f"figures/box_{index1}_{index2}{save_suffix}.png", dpi=400)

    if show:
        plt.show()
    else:
        plt.close()
