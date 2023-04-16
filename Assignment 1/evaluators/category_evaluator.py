from evaluators.evaluator import Evaluator

class CategoryEvaluator(Evaluator):

    def __init__(self, target, predictor):
        super().__init__(target, predictor)

    def evalutate(self, test_df):
        super().evalutate(test_df)

        if not test_df[self.target].dtype.name == 'category':
            raise Exception(f"{self.target} is not a categorical column.")

        correct_predictions = 0
        for i in range(len(self.test_df.index)):
            entity = self.test_df.iloc[[i]]

            prediction = self.predictor.predict(entity)
            actual = entity[self.target].values[0]

            correct_predictions += prediction == actual

        return correct_predictions / len(self.test_df.index)
