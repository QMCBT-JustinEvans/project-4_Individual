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

######################### Visualizations #########################

def viz_Q1(train):
    """
    Show visualization with minimal code in the workbook.
    """
    sns.lmplot(y='SEX_BTSX', x='life_expectancy', data=train, line_kws={'color': 'red'})
    sns.lmplot(y='SEX_MLE', x='life_expectancy', data=train, line_kws={'color': 'red'})
    sns.lmplot(y='SEX_FMLE', x='life_expectancy', data=train, line_kws={'color': 'red'})
    
def viz_Q2(train):
    sns.catplot(data=train, x="YEAR", 
                y="life_expectancy", 
                hue="SEX_MLE",
                kind="violin", 
                bw=.25, cut=0, split=True)

def viz_Q3(train):
    sns.catplot(data=train, y='life_expectancy', x='YEAR')
    
def viz_Q4(train):
    ax = sns.boxplot(x='YEAR', y='life_expectancy', data=train, color='#99c2a2')
    ax = sns.swarmplot(x="YEAR", y="life_expectancy", data=train, color='#7d0013', alpha=0.5)
    
######################### Statistical Tests #########################
    
def ttest(df, feature_1, feature_2, tail):
    """
    Conduct T-Test
    """

    t_stat, p_val = stats.levene(df[feature_1], df[feature_2])

    # Set local environment
    α = Alpha = alpha = 0.05
    random_state = 1992
    np.random.seed(1992)
    
    if p_val < α:
        print('equal_var = False (Equal Variance cannot be assumed)')
        #Using Scipy 
        t_stat, p_val = stats.ttest_ind(df[feature_1], 
                                        df[feature_2], 
                                        equal_var = False)
        print('_______________________________________________________________')  
        print(f't-stat: {t_stat}')
        print(f'p-value: {p_val}')

    else:
        print('equal_var = True (Equal Variance can be assumed)')
        #Using Scipy 
        t_stat, p_val = stats.ttest_ind(df[feature_1], 
                                        df[feature_2], 
                                        equal_var = True)
        print('_______________________________________________________________')  
        print(f't-stat: {t_stat}')
        print(f'p-value: {p_val}')

    print('_______________________________________________________________')  
        
    if tail == 1:
        # one_tail
        if (t_stat > 0) and ((p_val / 2) < α):
            print('Reject the null hypothesis')
        else:
            print('Fail to reject the null hypothesis')
        
    if tail == 2:
        # two_tail
        if p_val < α:
            print('Reject the null hypothesis')
        else:
            print('Fail to reject the null hypothesis')
            
def anova_test(df, feature_1, feature_2):
    """
    Conduct Anova Test
    """
    
    # stats f_oneway functions takes the groups as input and returns ANOVA F and p value
    f_val, p_val = stats.f_oneway(df[feature_2], df[feature_1])

    print(f'f_val: {f_val}')
    print(f'p_val: {p_val}')
    
    print('_______________________________________________________________')  

    if p_val < α:
        print('Reject the null hypothesis')
    else:
        print('Fail to reject the null hypothesis')