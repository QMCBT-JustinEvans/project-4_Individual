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

# import sklearn.metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import plot_confusion_matrix 
from sklearn.metrics import classification_report 
from sklearn.metrics import mean_squared_error, r2_score

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

######################### ACQUIRE DATA #########################

def get_leam():
    """
    This function pulls in the CSV as a DataFrame
    
    Due to latency and timeout issues, the CSV Data Pull weblink above must be used to pull the CSV directly.
    The csv should be saved to a local file. Multiple refresh attempts may be required before getting the save file prompt.
    
    GHO Selections: https://apps.who.int/gho/athena/api/GHO
    Homepage: https://www.who.int/data/gho/info/athena-api-examples
    CSV Data Pull: https://apps.who.int/gho/athena/api/GHO/WHOSIS_000001,WHOSIS_000002,WHOSIS_000007,WHOSIS_000015?format=csv
    """
    
    df = pd.read_csv('leam.csv')

    return df

######################### PREPARE DATA #########################

def clean_leam(df):

    """
    This function is used to clean the Life Expectancy And Mortaloty (LEAM) data as needed 
    ensuring not to introduce any new data but only remove irrelevant data 
    or reshape existing data to useable formats.
    """
    
    # Drop all columns with more than 10k Null and uneeded features
    df = df.drop(columns=['Low',
                          'High',
                          'StdErr', 
                          'StdDev', 
                          'Comments', 
                          'WORLDBANKINCOMEGROUP'
                         ])
    
    # IMPUTE NaN for [COUNTRY] with 'GLOBAL' if [REGION] == 'GLOBAL'
    missing_mask = df['COUNTRY'].isna()
    mapping_dict = dict({'GLOBAL': 'GLOBAL'})
    df.loc[missing_mask, 'COUNTRY'] = df.loc[missing_mask, 'REGION'].map(mapping_dict)
    
    df = df.dropna()
    
    # Create pivot table
    leam_pivot = df.pivot_table(index=['YEAR','COUNTRY','SEX'], columns='GHO', values='Numeric')
    
    # assign pivot to df and reset index
    df = leam_pivot.reset_index()
    
    # Create life_expectancy feature consisting of the mean of individual life expectancy features
    df['life_expectancy'] = df[['WHOSIS_000001','WHOSIS_000002','WHOSIS_000007','WHOSIS_000015']].mean(axis=1)
    
    # Use pandas dummies to pivot features with more than two string values
    #     into multiple columns with binary int values that can be read as boolean
    dummy_df = pd.get_dummies(data=df[['SEX']], drop_first=False)

    # Concat to leam DataFrame
    df = pd.concat([df, dummy_df], axis=1)
    
    # DROP original row for redundancy
    df = df.drop(columns=['SEX'])
    
    # Cache a Clean version of my data
    df.to_csv('clean_leam.csv')
        
    return df

        

######################### SPLIT DATA #########################

def split(df, stratify=False, target=None):
    """
    This Function splits the DataFrame into train, validate, and test
    then prints a graphic representation and a mini report showing the shape of the original DataFrame
    compared to the shape of the train, validate, and test DataFrames.
    
    IMPORTS Required:
    from sklearn.model_selection import train_test_split
    
    ARGUMENTS:
          df - Input the DataFrame you will split
    stratify - True will stratify for your Target (Do NOT stratify on continuous data)
               False will ignore this function
      target - Only needed if you will stratify
    """
    
    # Do NOT stratify on continuous data
    if stratify:
        # Split df into train and test using sklearn
        train, test = train_test_split(df, test_size=.2, random_state=1992, stratify=df[target])
        # Split train_df into train and validate using sklearn
        train, validate = train_test_split(train, test_size=.25, random_state=1992, stratify=df[target])
        
    else:
        train, test = train_test_split(df, test_size=.2, random_state=1992)
        train, validate = train_test_split(train, test_size=.37, random_state=1992)
    
    # reset index for train validate and test
    train.reset_index(drop=True, inplace=True)
    validate.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    train_prcnt = round((train.shape[0] / df.shape[0]), 2)*100
    validate_prcnt = round((validate.shape[0] / df.shape[0]), 2)*100
    test_prcnt = round((test.shape[0] / df.shape[0]), 2)*100
    
    print('________________________________________________________________')
    print('|                              DF                              |')
    print('|--------------------:--------------------:--------------------|')
    print('|        Train       |      Validate      |        Test        |')
    print(':--------------------------------------------------------------:')
    print()
    print()
    print(f'Prepared df: {df.shape}')
    print()
    print(f'      Train: {train.shape} - {train_prcnt}%')
    print(f'   Validate: {validate.shape} - {validate_prcnt}%')
    print(f'       Test: {test.shape} - {test_prcnt}%')
 
    
    return train, validate, test

def viz_split(train, validate, test):
    plt.figure(figsize=(12, 4))
    plt.title('Distribution of train, validate, and test')
    plt.plot(train.index, train.life_expectancy, color='lightgreen')
    plt.plot(validate.index, validate.life_expectancy, color='goldenrod')
    plt.plot(test.index, test.life_expectancy, color='blue', alpha=.5)
    plt.legend(['train', 'validate', 'test'], fontsize=15)
    
    return plt.show()

def Xy_split(feature_cols, target, train, validate, test):
    """
    This function will split the train, validate, and test data by the Feature Columns selected and the Target.
    
    Imports Needed:
    from sklearn.model_selection import train_test_split
    
    Arguments Taken:
       feature_cols: list['1','2','3'] the feature columns you want to run your model against.
             target: list the 'target' feature that you will try to predict
              train: Assign the name of your train DataFrame
           validate: Assign the name of your validate DataFrame
               test: Assign the name of your test DataFrame
    """
    
    print('_______________________________________________________________')
    print('|                              DF                             |')
    print('|-------------------:-------------------:---------------------|')
    print('|       Train       |       Validate    |          Test       |')
    print('|-------------------:-------------------:---------------------|')
    print('| x_train | y_train |   x_val  |  y_val |   x_test  |  y_test |')
    print(':-------------------------------------------------------------:')
    
    X_train, y_train = train[feature_cols], train[target]
    X_val, y_val = validate[feature_cols], validate[target]
    X_test, y_test = test[feature_cols], test[target]

    print()
    print()
    print(f'   X_train: {X_train.shape}   {X_train.columns}')
    print(f'   y_train: {y_train.shape}     Index({target})')
    print()
    print(f'X_validate: {X_val.shape}   {X_val.columns}')
    print(f'y_validate: {y_val.shape}     Index({target})')
    print()
    print(f'    X_test: {X_test.shape}   {X_test.columns}')
    print(f'    y_test: {y_test.shape}     Index({target})')
    
    
    return X_train, y_train, X_val, y_val, X_test, y_test

