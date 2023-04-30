import math
import numpy as np

from evaluators.evaluator import Evaluator
from predictors.numerical_predictor import NumericalPredictor

class RecommenderEvaluator(Evaluator):

    def __init__(self, target, predictor, groupby_column):
        super().__init__(target, predictor)

        # Recommender evaluators only work with numerical predictors
        if not isinstance(predictor, NumericalPredictor):
            raise TypeError("Predictor is not a numerical predictor")

        self.groupby_column = groupby_column

    def evaluate(self, test_df):

        metrics = []

        # Compute the metric over all groups
        for column_name, group in test_df.groupby(self.groupby_column):

            if len(group.index) == 0:
                continue

            metrics.append(self.compute_metric(group))

        return np.mean(metrics)

    def compute_metric(self, group):

        indices, predictions = self.get_predictions(group)

        # Sort indices based on predictions
        sorted_ = sorted(list(zip(indices, predictions)), key=lambda x: x[1], reverse=True)
        indices_sorted = [index for index, prediction in sorted_]

        # Get the actual values
        target_values = [group.loc[index, self.target] for index in indices_sorted]

        return self.calc_discounted_cumulative_gain(target_values)

    def get_predictions(self, group):
        
        indices, predictions = [], []
        for index, row in group.iterrows():

            prediction = self.predictor.predict(row)

            indices.append(index)
            predictions.append(prediction)

        return indices, predictions

    def calc_discounted_cumulative_gain(self, values):

        total = 0
        for i, value in enumerate(values):

            total += (2**value - 1) / (math.log2(i + 2))

        return total
    