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

import Preprocessing as pp
import Guess as gs
import KNN as KNN_regressor
import Linear_SGD as LR_classifier
import Random_forest as RF_regressor
import Regression_trees as RT_classifier

# Importing the dataset
dataset = pd.read_csv('health_insurance_train.csv')
dataset = pd.DataFrame(dataset)

# Preprocessing the dataset
X, y = pp.Pipeline_1(dataset)

# Making a guess
#mean, median, mae_mean, mae_median = gs.make_guess(dataset)

# Training and evaluating the KNN regression model with default hyperparameters
knn_model, knn_mae = KNN_regressor.train_and_evaluate_KNN_default(X, y)
print(knn_mae)

# Training and evaluating the Linear Regression model with default hyperparameters
sgd_model, sgd_mae = LR_classifier.train_and_evaluate_SGD_default(X, y)
print(sgd_mae)