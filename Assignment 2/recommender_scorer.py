import math
import numpy as np
import pandas as pd

class RecommenderScorer:

    def __init__(self, input_path, test_df, score_column):

        self.input_path = input_path
        self.test_df = test_df

        self.recommendation_df = pd.read_csv(self.input_path, sep=",")
    
        self.groupby_column = self.recommendation_df.columns[0]
        self.target_column = self.recommendation_df.columns[1]
        self.score_column = score_column

    def score(self):

        metrics = []

        groups = self.recommendation_df.groupby(self.groupby_column)
        test_groups = self.test_df.groupby(self.groupby_column)
        
        # Compute the metric over all groups
        for i, (column_name, group) in enumerate(groups):

            if len(group.index) == 0:
                continue

            metrics.append(self.compute_metric(group, test_groups.get_group(column_name)))

        return np.mean(metrics)

    def compute_metric(self, group, test_group):

        score_values = group.apply(lambda row:
            self.get_score_attribute(test_group, row),
            axis=1
        )

        return self.calc_discounted_cumulative_gain(score_values)

    def get_score_attribute(self, test_group, row):

        row_target_value = row[self.target_column]

        matching_row = test_group.loc[test_group[self.target_column] == row[self.target_column]]
        
        value = matching_row[self.score_column].values[0]
        return value

    def calc_discounted_cumulative_gain(self, values):

        total = 0
        for i, value in enumerate(values):

            total += (2**value - 1) / (math.log2(i + 2))

        return total
