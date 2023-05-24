import numpy as np
import pandas as pd

from validators.validator import Validator

class KFoldValidator(Validator):

    def __init__(self, df, evaluator, predictor, n_folds=5):
        super().__init__(df, evaluator, predictor)

        self.n_folds = n_folds

    def validate(self):
        super().validate()
        
        scores = []

        n_validation_rows = int(len(self.df.index) / self.n_folds)

        for i in range(self.n_folds):

            min_index = i * n_validation_rows
            max_index = (i + 1) * n_validation_rows

            validation_df = self.df[min_index:max_index]
            training_df = pd.concat([self.df[:min_index], self.df[max_index:]])

            self.predictor.train(training_df)

            score = self.evaluator.evaluate(validation_df)
            scores.append(score)
        
        return np.mean(scores), np.std(scores) / np.sqrt(self.n_folds)
