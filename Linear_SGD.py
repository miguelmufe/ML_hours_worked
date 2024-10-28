import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV, KFold, train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

def train_and_evaluate_SGD_default(X, y):

    # Initialize the SGD Regressor
    sgd_regressor = SGDRegressor(random_state = 0)
    
    # Evaluate the model using K-fold cross validation with K=5
    kf = KFold(n_splits=5, shuffle=True, random_state = 0)
    scores = cross_val_score(sgd_regressor, X, y, cv=kf, scoring='neg_mean_absolute_error')
    mae = -np.mean(scores)

    sgd_regressor.fit(X, y)
    
    return sgd_regressor, mae


def train_and_evaluate_SGD_tuned(X, y):
    # Define the parameter grid
    param_grid = {
        'max_iter': [50, 100, 200, 400, 800, 1600, 3200],
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
        'eta0': [0.001, 0.01, 0.1]
    }

    # Initialize the SGD Regressor
    sgd_regressor = SGDRegressor(random_state = 0)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=sgd_regressor, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)

    # Fit GridSearchCV
    grid_search.fit(X, y)

    # Get the best estimator
    best_sgd_regressor = grid_search.best_estimator_
    
    best_params = grid_search.best_params_
    
    # Initialize lists to store MAE per epoch
    train_mae_per_epoch = []
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    epochs = best_params['max_iter']
    

    # Train the model from scratch at each epoch using best params
    for epoch in range(1, epochs + 1):
        epoch_mae_scores = []
        
        # For each fold in KFold, train and evaluate
        for train_index, val_index in cv.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            
            # Reset the model with best params and fit on current training fold
            epoch_model = SGDRegressor(random_state=0, warm_start=True, max_iter=epoch, alpha = best_params['alpha'], 
                                       learning_rate = best_params['learning_rate'], eta0 = best_params['eta0'])
            epoch_model.fit(X_train, y_train)
            
            # Calculate MAE on the validation set
            y_val_pred = epoch_model.predict(X_val)
            epoch_mae_scores.append(mean_absolute_error(y_val, y_val_pred))
        
        # Average MAE across folds for the current epoch
        mean_epoch_mae = np.mean(epoch_mae_scores)
        train_mae_per_epoch.append(mean_epoch_mae)
    # Plot the training curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_mae_per_epoch, label='Cross-validated MAE', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('Training Curve (Cross-validated MAE vs. Epochs)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Final cross-validated MAE for the best model
    final_mae = train_mae_per_epoch[-1]
    print("Best hyperparameters for SGD:", grid_search.best_params_)

    return grid_search.best_estimator_, final_mae