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



def train_and_evaluate_KNN_default(X, y):

    # Training the KNN regression model using K-fold cross validation with K=5
    knn_regressor = KNeighborsRegressor()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(knn_regressor, X, y, cv = kf, scoring='neg_mean_absolute_error')

    # Print the cross-validation scores
    #print(f"Mean MSE: {-np.mean(cv_scores):.3f}, Standard Deviation of MSE: {np.std(cv_scores):.3f}")
    
    return knn_regressor, -np.mean(cv_scores)