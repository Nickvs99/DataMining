import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import scipy.optimize

import errors
from predictors.numerical_predictor import NumericalPredictor

class RegressionPredictor(NumericalPredictor):

    def __init__(self, target):
        super().__init__(target)
    
    def train(self, training_df):
        super().train(training_df)

        weights = self.init_weights()

        actual_values = []
        for i, entity in self.training_df.iterrows():
            actual_values.append(entity[self.target])

        # Minimize the error function by updating the weights
        response = scipy.optimize.minimize(self.minimize, weights, args=(actual_values,))
        self.weights = response.x

    def init_weights(self):
        
        self.weight_labels = ["Bias"]

        # Start with 1 weight, due to the bias term
        n_weights = 1
        for column in self.training_df.columns:

            if column == self.target:
                continue

            if is_numeric_dtype(self.training_df[column].dtype):
                n_weights += 1
                self.weight_labels.append(column)

            elif self.training_df[column].dtype.name == 'category':
                n_weights += len(self.training_df[column].cat.categories)

                for category in self.training_df[column].cat.categories:
                    self.weight_labels.append(f"{column} - {category}")
            else:
                raise Exception("Unsopperted column type")
            
        # Initialize weights with a value of 1
        return np.full((n_weights,), 1)

    def predict(self, entity, weights=None):

        if weights is None:
            weights = self.weights

        # Start with the bias term
        weight_index = 0
        value = weights[weight_index]
        for i, column in enumerate(self.training_df.columns):

            if column == self.target:
                continue

            if is_numeric_dtype(self.training_df[column].dtype):
                weight_index += 1
                value += self.regression_func(entity[column], weights[weight_index])
                

            elif self.training_df[column].dtype.name == 'category':

                for category in self.training_df[column].cat.categories:
                    
                    weight_index += 1
                    if category == entity[column]:
                        value += self.regression_func(1, weights[weight_index])

        
        return value


    def minimize(self, weights, actual_values):

        prediction_values = []
        for i, entity in self.training_df.iterrows():
            prediction_values.append(self.predict(entity, weights))   

        return self.total_error(actual_values, prediction_values)  

    def total_error(self, actual_values, prediction_values):

        total = 0
        for actual, prediction in zip(actual_values, prediction_values):
            total += self.error_func(actual, prediction)

        return total / len(actual_values)

    def error_func(self, actual, prediction):
        return errors.MSE(actual, prediction)
    
    def regression_func(self, x, w):
        return w * x

    def get_weight_df(self):

        weight_df = pd.DataFrame(list(zip(self.weight_labels, self.weights)), columns=["Label", "Value"])

        return weight_df