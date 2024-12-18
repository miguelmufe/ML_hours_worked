import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def Pipeline_1(dataset):

    if 'whrswk' in dataset.columns:
        dataset_X = dataset.drop(columns=['whrswk'])
        y = dataset['whrswk'].values
    else:
        dataset_X = dataset
        y = None

    # Drop missing values in both dataset_X and y
    if y is not None:
        non_missing_indices = dataset_X.dropna().index
        dataset_X = dataset_X.loc[non_missing_indices]
        y = y[non_missing_indices]
    else:
        dataset_X.dropna(inplace=True)

    # One-Hot Encoding for categorical features
    categorical_features = dataset_X.select_dtypes(include=['object']).columns
    dataset_X = pd.get_dummies(dataset_X, columns=categorical_features, drop_first=True)
    
    X = dataset_X.values
    
    return X, y

def Pipeline_2(dataset):

    if 'whrswk' in dataset.columns:
        dataset_X = dataset.drop(columns=['whrswk'])
        y = dataset['whrswk'].values
    else:
        dataset_X = dataset
        y = None

    # Apply SimpleImputer only to numeric columns
    numeric_features = dataset_X.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy='mean')
    dataset_X[numeric_features] = imputer.fit_transform(dataset_X[numeric_features])

    # One-Hot Encoding for categorical features
    categorical_features = dataset_X.select_dtypes(include=['object']).columns
    dataset_X = pd.get_dummies(dataset_X, columns=categorical_features, drop_first=True)

    # Feature Scaling
    scaler = StandardScaler()
    dataset_X = pd.DataFrame(scaler.fit_transform(dataset_X), columns=dataset_X.columns)

    X = dataset_X.values
    
    return X, y, scaler

