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
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


def train_and_evaluate_KNN_default(X, y):

    # Initialize the KNN Regressor
    knn_regressor = KNeighborsRegressor()

    # Evaluate the model using K-fold cross validation with K=5
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(knn_regressor, X, y, cv = kf, scoring='neg_mean_absolute_error')
    mae = -np.mean(scores)

    # Train the model
    knn_regressor.fit(X, y)

    return knn_regressor, mae

def train_and_evaluate_KNN_tuned(X, y):
    # Define the parameter grid
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }

    # Initialize the KNN Regressor
    knn_regressor = KNeighborsRegressor()

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=knn_regressor, param_grid=param_grid, 
                               scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)

    # Fit the model
    grid_search.fit(X, y)

    # Get the best estimator
    best_knn_regressor = grid_search.best_estimator_

    # Evaluate the best model using K-fold cross validation with K=5
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(best_knn_regressor, X, y, cv=kf, scoring='neg_mean_absolute_error')
    mae = -np.mean(scores)

    return best_knn_regressor, mae