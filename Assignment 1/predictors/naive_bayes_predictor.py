import random

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from predictors.category_predictor import CategoryPredictor

class NaiveBayesPredictor(CategoryPredictor):

    def __init__(self, target, n_category_bins=5):

        super().__init__(target)

        self.n_category_bins = n_category_bins

    def train(self, training_df):
        
        # Create a copy of the training_df, since NB needs adjustments to make
        # the algorithm work. Numerical columns have to be converted to categorical
        super().train(training_df.copy())

        # Convert numerical columns to categorical
        self.category_treshold_df = self.get_category_treshold_df()
        self.convert_numerical_to_categorical()

        self.target_probability_dict = self.get_target_probability_dict()

        # Compute table4.2 from DMlecture2
        # Structure -> key1: column, key2: column_category, key3: target_category
        self.probability_dict = self.compute_probability_dict()

    def predict(self, entity):

        # Compute likelihood for each category of the target and pick the largest

        target_categories = self.training_df[self.target].cat.categories
        likelihoods = [self.calc_likelihood(entity, target_category) for target_category in target_categories]

        max_likelihood = max(likelihoods)

        indices = [i for i, likelihood in enumerate(likelihoods) if likelihood == max_likelihood]
        index = random.choice(indices)

        return target_categories[index]


    def get_category_treshold_df(self):
        """
        This df stores the values at which a new category starts. This is used to map
        numerical values to a categorical value.
        """

        category_treshold_df = pd.DataFrame()

        quantiles = [i / self.n_category_bins for i in range(1, self.n_category_bins + 1)]
        
        numerical_columns = self.training_df.select_dtypes(include=np.number)
        for column in numerical_columns:

            series = self.training_df[column]
            quantile_values = [series.quantile(quantile) for quantile in quantiles]
            
            category_treshold_df[column] = quantile_values

        return category_treshold_df

    def convert_numerical_to_categorical(self):

        numerical_columns = self.training_df.select_dtypes(include=np.number)
        for column in numerical_columns:

            series = self.training_df[column]

            categories = [self.get_category(column, value) for value in series]

            self.training_df[column] = categories
            self.training_df[column] = self.training_df[column].astype("category")
            
    def get_category(self, column, value):

        if pd.isna(value):
            return None

        tresholds = self.category_treshold_df[column].values

        for i, treshold in enumerate(tresholds):

            if treshold > value:
                return i

        # The value can be larger than the largest treshold when validation data contains
        # a value larger than any value in the training_data. In this case the value belongs
        # the last category
        return i
    
    def compute_probability_dict(self):

        probability_dict = {}
  
        for column in self.training_df.columns:

            if column == self.target:
                continue

            probability_dict[column] = {}

            column_categories = self.training_df[column].cat.categories
            n_column_categories = len(column_categories)

            for target_category in self.target_probability_dict.keys():
                
                # Count how many times each category in column has the target category in the target column
                # https://stackoverflow.com/a/70156246/12132063
                count = self.training_df[self.training_df[self.target] == target_category][column].value_counts()

                # Apply laplace estimator
                target_count = self.target_probability_dict[target_category] * len(self.training_df.index)
                probabilities = count.apply(self.laplace_estimator, args=(n_column_categories, target_count))
                
                # Append probabilities to the probability dict
                for column_category in column_categories:

                    if column_category not in probability_dict[column]:
                        probability_dict[column][column_category] = {}

                    probability_dict[column][column_category][target_category] = probabilities[column_category] 

        return probability_dict

    def get_target_probability_dict(self):
        target_counts = self.training_df[self.target].value_counts(normalize = True)
        return target_counts.to_dict()

    def laplace_estimator(self, count, n_column_categories, target_count):
        return (count + 1) / (target_count + n_column_categories)

    def calc_likelihood(self, entity, target_category):

        likelihood = 1

        for column in entity.columns:

            if column == self.target:
                factor = self.target_probability_dict[target_category]

            else:

                # Skip invalid values
                if pd.isna(entity[column].values[0]):
                    continue

                if is_numeric_dtype(entity[column]):

                    value = entity[column].values[0]
                    category = self.get_category(column, value)
                else:
                    category = entity[column].values[0]

                factor = self.probability_dict[column][category][target_category]

            likelihood *= factor

        return likelihood
