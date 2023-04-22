from pandas.api.types import is_numeric_dtype

import errors
from evaluators.evaluator import Evaluator

class NumericalEvaluator(Evaluator):

    def __init__(self, target, predictor):
        super().__init__(target, predictor)
    
    def evaluate(self, test_df):
        super().evaluate(test_df)

        if not is_numeric_dtype(test_df[self.target]):
            raise Exception(f"{self.target} is not a numerical column")

        self.prediction_values, self.actual_values = [], []

        total_error = 0
        for i, entity in test_df.iterrows():

            prediction = self.predictor.predict(entity)
            actual = entity[self.target]

            total_error += self.error_func(actual, prediction)

            self.prediction_values.append(prediction)
            self.actual_values.append(actual)

        return total_error / len(test_df.index)
    
    def error_func(self, actual, prediction):
        return errors.MSE(actual, prediction)
