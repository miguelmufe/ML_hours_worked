import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV


def train_and_evaluate_RT_default(X, y):

    # Initialize the regression tree model
    rt_regressor = DecisionTreeRegressor()

    # Training the model using K-fold cross validation with K=5
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(rt_regressor, X, y, cv=kf, scoring='neg_mean_absolute_error')
    mae = -np.mean(scores)
    
    # Train the model
    rt_regressor.fit(X, y)

    return rt_regressor, mae

def train_and_evaluate_RT_tuned(X, y):
    # Initialize the regression tree model
    rt_regressor = DecisionTreeRegressor(random_state=42)

    # Define the parameter grid
    param_grid = {
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [None, 'auto', 'sqrt', 'log2']
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(rt_regressor, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)

    # Fit the model
    grid_search.fit(X, y)

    # Get the best model
    best_rt_regressor = grid_search.best_estimator_

    # Calculate the mean absolute error using cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(best_rt_regressor, X, y, cv=kf, scoring='neg_mean_absolute_error')
    mae = -np.mean(scores)

    return best_rt_regressor, mae