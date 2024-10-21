import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score

# Importing the dataset
dataset = pd.read_csv('health_insurance_train.csv')

# Pipeline1: 
X_1 = dataset.drop(columns=['whrswk']).values
y_1 = dataset['whrswk'].values
print("Pipeline1: ")
print(f"Shape of X_1: {np.shape(X_1)}")
print(f"Shape of y_1: {np.shape(y_1)}")

#Pipeline2:
X_2 = dataset.drop(columns=['whrswk']).values
y_2 = dataset['whrswk'].values
print("Pipeline2: ")
print(f"Shape of X_2: {np.shape(X_2)}")
print(f"Shape of y_2: {np.shape(y_2)}")


def make_guess(dataset):

    # Calculate the average and mean of the target variable
    mean_target = np.mean(dataset['whrswk'])
    median_target = np.median(dataset['whrswk'])

    # Create a numpy array with the same length as the dataset, filled with the average and mean values
    mean_array = np.full(len(dataset), mean_target)
    median_array = np.full(len(dataset), median_target)

    #Calculate the Mean Absolute Error for the average and mean values
    mae_mean = mean_absolute_error(y_1, mean_array)
    mae_median = mean_absolute_error(y_1, median_array)

    # Print the results
    print(f"Mean: {mean_array[0]}")
    print(f"Mean Absolute Error: {mae_mean}")
    print(f"Median: {median_array[0]}")
    print(f"Mean Absolute Error: {mae_median}")

    return mean_array[0], median_array[0], mae_mean, mae_median

mean, median, mae_mean, mae_median = make_guess(dataset)

