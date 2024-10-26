
from sklearn.metrics import mean_absolute_error
import numpy as np


def make_guess(y):

    # Calculate the average and mean of the target variable
    mean_target = np.mean(y)
    median_target = np.median(y)

    # Create a numpy array with the same length as the dataset, filled with the average and mean values
    mean_array = np.full(len(y), mean_target)
    median_array = np.full(len(y), median_target)

    #Calculate the Mean Absolute Error for the average and mean values
    mae_mean = mean_absolute_error(y, mean_array)
    mae_median = mean_absolute_error(y, median_array)    

    return mean_target, median_target, mae_mean, mae_median