import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

def knn(target, training_df, validation_df, k=5):

    correct_predictions = 0
    for i in range(len(validation_df.index)):
        instance = validation_df.iloc[[i]]

        prediction = predict_class(target, instance, training_df, k=k)
        actual = instance[target].values[0]

        correct_predictions += prediction == actual

    return correct_predictions / len(validation_df.index)

def predict_class(target, instance, training_df, k=5):

    nearest_neigbours = find_k_nearest_neighbours(target, instance, training_df, k=k)

    # Count how many times each category is in the nearest neighbours
    category_frequency = {}
    for i in range(len(nearest_neigbours)):

        neighbour = nearest_neigbours.iloc[[i]]
        category = neighbour[target].values[0]

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
    valid_neighbours =  nearest_neigbours[nearest_neigbours[target].isin(selected_categories)]

    # Since the nearest neighbours are sorted, we can simply return the instance at index 0
    return valid_neighbours.iloc[[0]][target].values[0]


def find_k_nearest_neighbours(target, instance, training_df, k=5):
    
    distances = []
    indices = []

    instance_index = instance.index.values[0]

    # Compute all distances between the instance and the training data
    for i in range(len(training_df.index)):

        row = training_df.iloc[[i]]
        index = row.index.values[0]

        if index == instance_index:
            continue
        
        distance = calc_distance(target, instance, row)

        distances.append(distance)
        indices.append(index)

    distances = np.array(distances)
    indices = np.array(indices)


    # Sort the indices based on the distance
    sort_ind = np.argsort(distances)
    sorted_ids = indices[sort_ind]

    # Select the best indices
    selected_ids = sorted_ids[:k]

    # Return the best neighbours based on the indices
    return training_df.loc[selected_ids, :]


def calc_distance(target, row1, row2, n=1):
    """
    Calculates the distance between two entries.

    n is a hyperparameter controlling the distance measure. 
    n = 1 --> manhattan distnace
    n = 2 --> euclidian distance
    """

    distance = 0
    for column in row1.columns:

        # The target column should not be taken into account in the distance measure
        if column == target:
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
            distance += abs(value1 - value2) ** n
        elif row1[column].dtype.name == 'category':
            distance += (value1 != value2) ** n
        else:
            print("Unsupported column type")

    return distance ** (1/n)

