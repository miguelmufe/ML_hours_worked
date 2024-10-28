from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import GridSearchCV


def train_and_evaluate_SGD_default(X, y):

    # Initialize the SGD Regressor
    sgd_regressor = SGDRegressor(random_state = 0)
    
    # Evaluate the model using K-fold cross validation with K=5
    kf = KFold(n_splits=5, shuffle=True, random_state = 0)
    scores = cross_val_score(sgd_regressor, X, y, cv=kf, scoring='neg_mean_absolute_error')
    mae = -np.mean(scores)

    sgd_regressor.fit(X, y)
    # Store the scaler for future use (e.g., transforming new data)
    return sgd_regressor, mae


def train_and_evaluate_SGD_tuned(X, y):
    # Define the parameter grid
    param_grid = {
        'loss': ['squared_loss', 'huber', 'epsilon_insensitive'],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
        'eta0': [0.01, 0.1, 1]
    }

    # Initialize the SGD Regressor
    sgd_regressor = SGDRegressor(random_state = 0)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=sgd_regressor, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)

    # Fit GridSearchCV
    grid_search.fit(X, y)

    # Get the best estimator
    best_sgd_regressor = grid_search.best_estimator_

    # Evaluate the best model using K-fold cross validation with K=5
    kf = KFold(n_splits=5, shuffle=True, random_state = 0)
    scores = cross_val_score(best_sgd_regressor, X, y, cv=kf, scoring='neg_mean_absolute_error')
    mae = -np.mean(scores)

    best_sgd_regressor.fit(X, y)
    # Store the scaler for future use (e.g., transforming new data)
    return best_sgd_regressor, mae