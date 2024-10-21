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
from sklearn.utils import resample

def Pipeline_1(dataset):
    #Drop missing values
    dataset.dropna(inplace=True)

    # One-Hot Encoding for categorical features
    categorical_features = dataset.select_dtypes(include=['object']).columns
    dataset = pd.get_dummies(dataset, columns=categorical_features, drop_first=True)

    X_1 = dataset.drop(columns=['whrswk']).values
    y_1 = dataset['whrswk'].values
    return X_1, y_1


def Pipeline_2(dataset):
    X_2 = dataset.drop(columns=['whrswk']).values
    y_2 = dataset['whrswk'].values

    #note: maybe use a scaler here
    return X_2, y_2

