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




# Making predictions
def make_predictions(knn_model, sgd_model, rf_model, rt_model, pipeline = 2):

    dataset_test = pd.read_csv('health_insurance_autograde.csv')
    dataset_test = pd.DataFrame(dataset_test)
    if pipeline == 1:
        X, y = pp.Pipeline_1(dataset_test)
    elif pipeline == 2:
        X, y, scaler = pp.Pipeline_2(dataset_test)

    # Making predictions using the KNN model
    y_pred_knn = knn_model.predict(X)

    # Making predictions using the SGD model
    y_pred_sgd = sgd_model.predict(X)

    # Making predictions using the Random Forest model
    y_pred_rf = rf_model.predict(X)

    # Making predictions using the Regression Trees model
    y_pred_rt = rt_model.predict(X)

    return y_pred_knn, y_pred_sgd, y_pred_rf, y_pred_rt

# Plotting the predictions

def plot_predictions(y_pred_knn, y_pred_sgd, y_pred_rf, y_pred_rt):
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    plt.hist(y_pred_knn, bins=range(int(min(y_pred_knn)), int(max(y_pred_knn)) + 1), edgecolor='black')
    plt.title('KNN Predictions')
    plt.xlabel('Predicted Value')
    plt.ylabel('Frequency')

    plt.subplot(2, 2, 2)
    plt.hist(y_pred_sgd, bins=range(int(min(y_pred_sgd)), int(max(y_pred_sgd)) + 1), edgecolor='black')
    plt.title('SGD Predictions')
    plt.xlabel('Predicted Value')
    plt.ylabel('Frequency')

    plt.subplot(2, 2, 3)
    plt.hist(y_pred_rf, bins=range(int(min(y_pred_rf)), int(max(y_pred_rf)) + 1), edgecolor='black')
    plt.title('Random Forest Predictions')
    plt.xlabel('Predicted Value')
    plt.ylabel('Frequency')

    plt.subplot(2, 2, 4)
    plt.hist(y_pred_rt, bins=range(int(min(y_pred_rt)), int(max(y_pred_rt)) + 1), edgecolor='black')
    plt.title('Regression Trees Predictions')
    plt.xlabel('Predicted Value')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()
