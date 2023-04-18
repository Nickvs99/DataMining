import pandas as pd

class Validator:

    def __init__(self, df, evaluator, predictor):

        self.df = df
        self.evaluator = evaluator
        self.predictor = predictor

    def validate(self):
        
        self.actual_values = None
        self.prediction_values = None

    def compute_confusion_df(self):

        if self.prediction_values is None or self.actual_values is None:
            raise ValueError("prediction or actual values are not set. Run evaluate() first.")

        total_column_label = "Total"
        confusion_columns = self.df[self.evaluator.target].cat.categories.tolist()
        confusion_columns.append(total_column_label)

        confusion_df = pd.DataFrame(index=confusion_columns, columns=confusion_columns)
        confusion_df = confusion_df.fillna(0)
        
        for actual, prediction in zip(self.actual_values, self.prediction_values):
            
            confusion_df.loc[actual, prediction] += 1
            confusion_df.loc[actual, total_column_label] += 1
            confusion_df.loc[total_column_label, prediction] += 1

        return confusion_df
