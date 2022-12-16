######################### IMPORTS #########################

# Import standard libraries
import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# Decision Tree and Model Evaluation Imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree, export_text

# import sklearn.linear_model
from sklearn.linear_model import LassoLars
from sklearn.linear_model import TweedieRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV

from sklearn.cluster import KMeans

# import sklearn.metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import plot_confusion_matrix 
from sklearn.metrics import classification_report 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import explained_variance_score

# import sklearn.preprocessing
import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PolynomialFeatures

# Set local state
α = Alpha = alpha = 0.05
random_state=1992
np.random.seed(1992)

# Hides future deprecation warnings
import warnings
warnings.filterwarnings("ignore")   


################################ models ####################################################


################################ Evaluation ####################################################

#def calculate_mse(y_predicted):
#    return mean_squared_error(predictions.actual, y_predicted)

#predictions.apply(calculate_mse).sort_values()

def get_predictions():
    # Read in predictions file
    predictions = pd.read_csv('predictions.csv')
    
    # run MSE on predictions
    #calculate_mse = mean_squared_error(predictions.actual, y_predicted)

    #return predictions.apply(calculate_mse).sort_values()
    return predictions

def rmse_eval(y_train, y_val):
    """
    Run Automated RMSE Report
    """

    # We need y_train and y_validate to be dataframes to append the new columns with predicted values. 
    y_train = pd.DataFrame(y_train)
    y_val = pd.DataFrame(y_val)

    # 1. Predict target_pred_mean
    target_pred_mean = y_train['life_expectancy'].mean()
    y_train['target_pred_mean'] = target_pred_mean
    y_val['target_pred_mean'] = target_pred_mean

    # 2. compute target_pred_median
    target_pred_median = y_train['life_expectancy'].median()
    y_train['target_pred_median'] = target_pred_median
    y_val['target_pred_median'] = target_pred_median

    # 3. RMSE of target_pred_mean
    rmse_train = mean_squared_error(y_train.life_expectancy, y_train.target_pred_mean)**(1/2)
    rmse_val = mean_squared_error(y_val.life_expectancy, y_val.target_pred_mean)**(1/2)

    print("RMSE using Mean\nTrain/In-Sample: ", round(rmse_train, 2), 
          "\nValidate/Out-of-Sample: ", round(rmse_val, 2))

    # 4. RMSE of G3_pred_median
    rmse_train = mean_squared_error(y_train.life_expectancy, y_train.target_pred_median)**(1/2)
    rmse_val = mean_squared_error(y_val.life_expectancy, y_val.target_pred_median)**(1/2)

    print('___________________________________________________________________________________')
    print("RMSE using Median\nTrain/In-Sample: ", round(rmse_train, 2), 
          "\nValidate/Out-of-Sample: ", round(rmse_val, 2))

        # We need y_train and y_validate to be dataframes to append the new columns with predicted values. 
    
    # plot to visualize actual vs predicted. 
    plt.figure(figsize=(8, 6))
    plt.hist(y_train.life_expectancy, color='lightgrey', alpha=1, label="Actual Life Expectancy")
    plt.hist(y_train.target_pred_mean, bins=1, color='red', alpha=1, rwidth=100, label="Predicted Life Expectancy (Mean)")
    plt.hist(y_train.target_pred_median, bins=1, color='orange', alpha=1, rwidth=100, label="Predicted Life Expectancy (Median)")
    plt.xlabel("Life Expectancy (target)")
    plt.ylabel("Number of Observations")
    plt.ylim(0,262)
    plt.legend(loc=2)
    plt.show()

def get_metric_TEST():
    # Read in predictions file
    metric_TEST = pd.read_csv('metric_TEST.csv')
    
    # run MSE on predictions
    #calculate_mse = mean_squared_error(predictions.actual, y_predicted)

    #return predictions.apply(calculate_mse).sort_values()
    return metric_TEST

def rmse_TEST_eval(y_train, y_val, y_test):
    """
    Run Automated RMSE Report
    """

    # We need y_train and y_validate to be dataframes to append the new columns with predicted values. 
    y_train = pd.DataFrame(y_train)
    y_val = pd.DataFrame(y_val)
    y_test = pd.DataFrame(y_test)
    
    # 1. Predict target_pred_mean
    target_pred_mean = y_test['life_expectancy'].mean()
    y_train['target_pred_mean'] = target_pred_mean
    y_val['target_pred_mean'] = target_pred_mean
    y_test['target_pred_mean'] = target_pred_mean

    
    # 2. compute target_pred_median
    target_pred_median = y_test['life_expectancy'].median()
    y_train['target_pred_median'] = target_pred_median
    y_val['target_pred_median'] = target_pred_median
    y_test['target_pred_median'] = target_pred_median
    
    # 3. RMSE of target_pred_mean
    rmse_train = mean_squared_error(y_train.life_expectancy, y_train.target_pred_mean)**(1/2)
    rmse_val = mean_squared_error(y_val.life_expectancy, y_val.target_pred_mean)**(1/2)
    rmse_test = mean_squared_error(y_test.life_expectancy, y_test.target_pred_mean)**(1/2)

    print("RMSE using Mean\nTrain/In-Sample: ", round(rmse_train, 2),
      "\nValidate/Out-of-Sample: ", round(rmse_val, 2),
      "\nTEST/Out-of-Sample: ", round(rmse_test, 2))

    # 4. RMSE of G3_pred_median
    rmse_train = mean_squared_error(y_train.life_expectancy, y_train.target_pred_median)**(1/2)
    rmse_val = mean_squared_error(y_val.life_expectancy, y_val.target_pred_median)**(1/2)
    rmse_test = mean_squared_error(y_test.life_expectancy, y_test.target_pred_median)**(1/2)

    print('___________________________________________________________________________________')
    print("RMSE using Median\nTrain/In-Sample: ", round(rmse_train, 2), 
          "\nValidate/Out-of-Sample: ", round(rmse_val, 2),
          "\nTEST/Out-of-Sample: ", round(rmse_test, 2))

    # We need y_train and y_validate to be dataframes to append the new columns with predicted values. 
    
    # plot to visualize actual vs predicted. 
    plt.figure(figsize=(8, 6))
    plt.hist(y_test.life_expectancy, color='lightgrey', alpha=1, label="Actual Life Expectancy")
    plt.hist(y_test.target_pred_mean, bins=1, color='red', alpha=1, rwidth=100, label="Predicted Life Expectancy (Mean)")
    plt.hist(y_test.target_pred_median, bins=1, color='orange', alpha=1, rwidth=100, label="Predicted Life Expectancy (Median)")
    plt.xlabel("Life Expectancy (target)")
    plt.ylabel("Number of Observations")
    plt.ylim(0,92)
    plt.legend(loc=2)
    plt.show()




################################ Visualization ####################################################

def viz_predictions(y_train, y_val):
    # We need y_train and y_validate to be dataframes to append the new columns with predicted values. 
    y_train = pd.DataFrame(y_train)
    y_val = pd.DataFrame(y_val)
    
    # plot to visualize actual vs predicted. 
    plt.figure(figsize=(8, 6))
    plt.hist(y_train.life_expectancy, color='lightgrey', alpha=1, label="Actual Life Expectancy")
    plt.hist(y_train.target_pred_mean, bins=1, color='red', alpha=1, rwidth=100, label="Predicted Life Expectancy (Mean)")
    plt.hist(y_train.target_pred_median, bins=1, color='orange', alpha=1, rwidth=100, label="Predicted Life Expectancy (Median)")
    plt.xlabel("Life Expectancy (target)")
    plt.ylabel("Number of Observations")
    plt.ylim(0,262)
    plt.legend(loc=2)
    plt.show()