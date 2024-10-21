
from sklearn.metrics import mean_absolute_error
import numpy as np


def make_guess(dataset):

    # Calculate the average and mean of the target variable
    mean_target = np.mean(dataset['whrswk'])
    median_target = np.median(dataset['whrswk'])

    # Create a numpy array with the same length as the dataset, filled with the average and mean values
    mean_array = np.full(len(dataset), mean_target)
    median_array = np.full(len(dataset), median_target)

    #Calculate the Mean Absolute Error for the average and mean values
    y = dataset['whrswk'].values
    mae_mean = mean_absolute_error(y, mean_array)
    mae_median = mean_absolute_error(y, median_array)    

    print(f'Distribution of the target variable: ')
    print(f"Mean: {mean_target}")
    print(f"Mean Absolute Error: {mae_mean}")
    print(f"Median: {median_target}")
    print(f"Median Absolute Error: {mae_median}")

    return mean_array[0], median_array[0], mae_mean, mae_median