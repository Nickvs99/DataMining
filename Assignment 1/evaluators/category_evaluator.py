from evaluators.evaluator import Evaluator

class CategoryEvaluator(Evaluator):

    def __init__(self, target, predictor):
        super().__init__(target, predictor)

    def evaluate(self, test_df):
        super().evaluate(test_df)

        if not test_df[self.target].dtype.name == 'category':
            raise Exception(f"{self.target} is not a categorical column.")

        self.prediction_values, self.actual_values = [], []
        correct_predictions = 0
        for i in range(len(self.test_df.index)):
            entity = self.test_df.iloc[[i]]

            prediction = self.predictor.predict(entity)
            actual = entity[self.target].values[0]

            correct_predictions += prediction == actual

            self.prediction_values.append(prediction)
            self.actual_values.append(actual)

        return correct_predictions / len(self.test_df.index)
