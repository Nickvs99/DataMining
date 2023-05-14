import math
import numpy as np

from evaluators.evaluator import Evaluator
from predictors.numerical_predictor import NumericalPredictor

from logger import logger

class RecommenderEvaluator(Evaluator):

    def __init__(self, target, predictor, groupby_column):
        super().__init__(target, predictor)

        # Recommender evaluators only work with numerical predictors
        if not isinstance(predictor, NumericalPredictor):
            raise TypeError("Predictor is not a numerical predictor")

        self.groupby_column = groupby_column

    def evaluate(self, test_df):
        
        logger.status("Evaluating recommender")
        
        metrics = []

        groups = test_df.groupby(self.groupby_column)

        # Compute the metric over all groups
        for i, (column_name, group) in enumerate(groups):

            logger.progress(f"Evaluating groups {i / groups.ngroups * 100:.2f}%")
            
            if len(group.index) == 0:
                continue

            metrics.append(self.compute_metric(group))

        return np.mean(metrics)

    def compute_metric(self, group):

        target_values = self.get_sorted_attribute_values(group, self.target)
        return self.calc_discounted_cumulative_gain(target_values)

    def get_predictions(self, group):
        
        predictions = []
        
        for index in group.index:
            row = group.loc[[index]]

            prediction = self.predictor.predict(row)

            predictions.append(prediction)

        return group.index.values, predictions

    def get_sorted_attribute_values(self, group, attribute):
        """
        Returns the attribute values after sorting the instances by the target attribute
        """
        
        indices, predictions = self.get_predictions(group)

        # Sort indices based on predictions
        sorted_ = sorted(list(zip(indices, predictions)), key=lambda x: x[1], reverse=True)
        indices_sorted = [index for index, prediction in sorted_]

        # Get the actual values
        attribute_values = [group.loc[index, attribute] for index in indices_sorted]
        
        return attribute_values

    def calc_discounted_cumulative_gain(self, values):

        total = 0
        for i, value in enumerate(values):

            total += (2**value - 1) / (math.log2(i + 2))

        return total
    
    def write_kaggle_prediction(self, test_df, output_path="output.csv"):

        lines = ["srch_id,prop_id"]
        
        for srch_id, group in test_df.groupby("srch_id"):

            if len(group.index) == 0:
                continue
    
            prop_ids = self.get_sorted_attribute_values(group, "prop_id")

            for prop_id in prop_ids:
                lines.append(f"{srch_id},{prop_id}")

        
        with open(output_path, 'w') as f:
            for line in lines:
                f.write(f"{line}\n")
