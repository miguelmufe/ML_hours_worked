from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import numpy as np
from sklearn.model_selection import GridSearchCV


def train_and_evaluate_RF_default(X, y):

    # Initialize the Random Forest Classifier
    rf_regressor = RandomForestClassifier(random_state = 0)

    # Evaluate the model using K-fold cross validation with K=5
    kf = KFold(n_splits=5, shuffle=True, random_state = 0)
    scores = cross_val_score(rf_regressor, X, y, cv=kf, scoring='neg_mean_absolute_error')
    mae = -np.mean(scores)

    # Train the model
    rf_regressor.fit(X, y)
    
    return rf_regressor, mae

def train_and_evaluate_RF_tuned(X, y):
    # Initialize the Random Forest Classifier
    rf_regressor = RandomForestClassifier(random_state = 0)

    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, 
                                cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)

    # Fit the model
    grid_search.fit(X, y)

    # Get the best model
    best_rf_regressor = grid_search.best_estimator_

    # Evaluate the best model using K-fold cross validation with K=5
    kf = KFold(n_splits=5, shuffle=True, random_state = 0)
    scores = cross_val_score(best_rf_regressor, X, y, cv=kf, scoring='neg_mean_absolute_error')
    mae = -np.mean(scores)

    print(f'Best hyperparameters for random forest: {grid_search.best_params_}')
    return best_rf_regressor, mae