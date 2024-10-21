from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
import numpy as np

def train_and_evaluate_SGD_default(X, y):

    sgd_regressor = SGDRegressor()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(sgd_regressor, X, y, cv=kf, scoring='neg_mean_absolute_error')
    mae = -np.mean(scores)
    return sgd_regressor, mae