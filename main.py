import pandas as pd
import Preprocessing as pp
import Guess as gs
import KNN as KNN_regressor
import Linear_SGD as LR_classifier
import Random_forest as RF_regressor
import Regression_trees as RT_classifier
import Predictions as predictions

tune = False
write_file = False

# ---------------------Importing the dataset-----------------------------------------------
dataset = pd.read_csv('health_insurance_train.csv')
dataset = pd.DataFrame(dataset)

# ------------Training and evaluating the models with default hyperparameters-----------------------
def default_hyperparameters(dataset, pipeline):

    # Preprocessing the dataset
    if pipeline == 1:
        X, y = pp.Pipeline_1(dataset)
    elif pipeline == 2:
        X, y, scaler = pp.Pipeline_2(dataset)

    # Making a guess
    mean, median, mae_mean, mae_median = gs.make_guess(y)
    print(f'MAE for mean: {mae_mean}')
    print(f'MAE for median: {mae_median}')

    # Training and evaluating the KNN regression model with default hyperparameters
    knn_model, knn_mae = KNN_regressor.train_and_evaluate_KNN_default(X, y)
    print(f'MAE for KNN with default hyperparameters {knn_mae}')

    # Training and evaluating the SGD Linear Regression model with default hyperparameters
    sgd_model, sgd_mae = LR_classifier.train_and_evaluate_SGD_default(X, y)
    print(f'MAE for SGD with default hyperparameters {sgd_mae}')

    # Training and evaluating the Random Forest regression model with default hyperparameters
    rf_model, rf_mae = RF_regressor.train_and_evaluate_RF_default(X, y)
    print(f'MAE for Random Forest with default hyperparameters {rf_mae}')

    # Training and evaluating the Regression Trees classifier with default hyperparameters
    rt_model, rt_mae = RT_classifier.train_and_evaluate_RT_default(X, y)
    print(f'MAE for Regression Trees with default hyperparameters {rt_mae}')

    if pipeline == 1:
        return knn_model, sgd_model, rf_model, rt_model
    elif pipeline == 2:
        return knn_model, sgd_model, rf_model, rt_model, scaler

if tune == False:
    print('-------------------Pipeline 1 with default hyperparameters-------------------')
    knn_model_def_1, sgd_model_def_1, rf_model_def_1, rt_model_def_1= default_hyperparameters(dataset, 1)
    print('-------------------Pipeline 2 with default hyperparameters-------------------')
    knn_model_def_2, sgd_model_def_2, rf_model_def_2, rt_model_def_2, scaler= default_hyperparameters(dataset, 2)

# ---------------Training and evaluating the models with tuned hyperparameters-----------------
def tuned_hyperparameters(dataset, pipeline):

    # Preprocessing the dataset
    if pipeline == 1:
        X, y= pp.Pipeline_1(dataset)
    elif pipeline == 2:
        X, y, scaler = pp.Pipeline_2(dataset)
    
    # Training and evaluating the KNN regression model with default hyperparameters
    knn_model, knn_mae = KNN_regressor.train_and_evaluate_KNN_tuned(X, y)
    print(f'MAE for KNN with tuned hyperparameters {knn_mae}')
    
    # Training and evaluating the SGD Linear Regression model with default hyperparameters
    sgd_model, sgd_mae = LR_classifier.train_and_evaluate_SGD_tuned(X, y)
    print(f'MAE for SGD with tuned hyperparameters {sgd_mae}')

    # Training and evaluating the Random Forest regression model with default hyperparameters
    rf_model, rf_mae = RF_regressor.train_and_evaluate_RF_tuned(X, y)
    print(f'MAE for Random Forest with tuned hyperparameters {rf_mae}')
    
    # Training and evaluating the Regression Trees classifier with default hyperparameters
    rt_model, rt_mae = RT_classifier.train_and_evaluate_RT_tuned(X, y)
    print(f'MAE for Regression Trees with tuned hyperparameters {rt_mae}')
   
    if pipeline == 1:
        return knn_model, sgd_model, rf_model, rt_model
    elif pipeline == 2:
        return knn_model, sgd_model, rf_model, rt_model
if tune == True:
    knn_model_tuned, sgd_model_tuned, rf_model_tuned, rt_model_tuned= tuned_hyperparameters(dataset, 2)

# ------------------Making predictions on autograder and plotting the results-----------------------------
if tune == False:
    y_pred_knn_def_1, y_pred_sgd_def_1, y_pred_rf_def_1, y_pred_rt_def_1 = predictions.make_predictions(
        knn_model_def_1, sgd_model_def_1, rf_model_def_1, rt_model_def_1, pipeline=1)
    y_pred_knn_def_2, y_pred_sgd_def_2, y_pred_rf_def_2, y_pred_rt_def_2 = predictions.make_predictions(
        knn_model_def_2, sgd_model_def_2, rf_model_def_2, rt_model_def_2, pipeline=2)
    
    predictions.plot_predictions(y_pred_knn_def_2, y_pred_sgd_def_2, y_pred_rf_def_2, y_pred_rt_def_2)

    if write_file == True:
        with open('autograder_submission.txt', 'w') as file:
            file.write('11.9104\n')
            for pred in y_pred_rf_def_2:
                file.write(f'{pred}\n')

if tune == True:
    y_pred_knn_tuned, y_pred_sgd_tuned, y_pred_rf_tuned, y_pred_rt_tuned = predictions.make_predictions(
        knn_model_tuned, sgd_model_tuned, rf_model_tuned, rt_model_tuned)
    predictions.plot_predictions(y_pred_knn_tuned, y_pred_sgd_tuned, y_pred_rf_tuned, y_pred_rt_tuned)

    if write_file == True:
        with open('autograder_submission.txt', 'w') as file:
            file.write('11.9104\n')
            for pred in y_pred_rf_tuned:
                file.write(f'{pred}\n')