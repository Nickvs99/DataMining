import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from predictors.category_predictor import CategoryPredictor

class KnnPredictor(CategoryPredictor):

    def __init__(self, target, k=5, n=2):
        super().__init__(target)

        self.k = k
        self.n = n

    def predict(self, entity):
        
        nearest_neigbours = self.find_nearest_neighbours(entity)

        # Count how many times each category is in the nearest neighbours
        category_frequency = {}
        for i in range(len(nearest_neigbours)):

            neighbour = nearest_neigbours.iloc[[i]]
            category = neighbour[self.target].values[0]

            if category in category_frequency:
                category_frequency[category] += 1
            else:
                category_frequency[category] = 1

        # Sort the categories by their frequency in descending order
        category_frequency = dict(sorted(category_frequency.items(), key=lambda item: item[1], reverse=True))

        # Check if there are multiple categories with equal frequency
        max_frequency =  max(category_frequency.values())
        selected_categories = [category for category in category_frequency if category_frequency[category] == max_frequency]

        if len(selected_categories) == 1:
            return selected_categories[0]

        # In case of tie, return the closest neighbour which has a one of the tied categories
        valid_neighbours =  nearest_neigbours[nearest_neigbours[self.target].isin(selected_categories)]

        # Since the nearest neighbours are sorted, we can simply return the instance at index 0
        return valid_neighbours.iloc[[0]][self.target].values[0]


    def find_nearest_neighbours(self, entity):
        
        distances = []
        indices = []

        entity_index = entity.index.values[0]

        # Compute all distances between the instance and the training data
        for i in range(len(self.training_df.index)):

            row = self.training_df.iloc[[i]]
            index = row.index.values[0]

            if index == entity_index:
                continue
            
            distance = self.calc_distance(entity, row)

            distances.append(distance)
            indices.append(index)

        distances = np.array(distances)
        indices = np.array(indices)

        # Sort the indices based on the distance
        sort_ind = np.argsort(distances)
        sorted_ids = indices[sort_ind]

        # Select the k best indices
        selected_ids = sorted_ids[:self.k]

        # Return the best neighbours based on the indices
        return self.training_df.loc[selected_ids, :]


    def calc_distance(self, row1, row2):
        """
        Calculates the distance between two entries.

        n is a hyperparameter controlling the distance measure. 
        n = 1 --> manhattan distnace
        n = 2 --> euclidian distance
        """

        distance = 0
        for column in row1.columns:

            # The target column should not be taken into account in the distance measure
            if column == self.target:
                continue

            value1 = row1[column].values[0]
            value2 = row2[column].values[0]

            # Skip if the training test instace has a nan value
            if pd.isna(value1):
                continue
            
            # Maximum penalty if the training instance has a nan value
            elif pd.isna(value2):
                distance += 1

            elif is_numeric_dtype(row1[column].dtype):
                distance += abs(value1 - value2) ** self.n
            elif row1[column].dtype.name == 'category':
                distance += (value1 != value2) ** self.n
            else:
                print("Unsupported column type")

        return distance ** (1/self.n)

