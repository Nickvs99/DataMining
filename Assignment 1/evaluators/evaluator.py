

class Evaluator:

    def __init__(self, target, predictor):

        self.target = target
        self.predictor = predictor
        
    def evalutate(self, test_df):
        
        self.test_df = test_df

        if self.target not in self.test_df.columns:
            raise Exception(f"Invalid target. Possible values are {self.test_df.columns}.")

