#!/usr/bin/env python
# coding: utf-8

# In[34]:


print("**CUSTOM EXPLORATION FUNCTIONS")
print("nunique_column_all(df): PRINT NUNIQUE OF ALL COLUMNS")
print("nunique_column_objects(df): PRINT NUNIQUE OF COLUMNS THAT ARE OBJECTS")
print("nunique_column_qty(df): PRINT NUNIQUE OF COLUMNS THAT ARE *NOT* OBJECTS")
print("numeric_range(df): COMPUTE RANGE FOR ALL NUMERIC VARIABLES")
print("")
print("**USEFUL EXPLORATORY CODE**")
print ("DFNAME.head()")
print ("DFNAME.shape")
print ("DFNAME.shape[0] #read row count")
print ("DFNAME.describe().T")
print ("DFNAME.columns.to_list()")
print("DFNAME.COLUMNNAME.value_counts(dropna=False)")
print ("DFNAME.dtypes")
print("DFNAME.select_dtypes(include='object').columns")
print("DFNAME.select_dtypes(include='float').columns")
print("pd.crosstab(DFNAME.COLUMN-1, DFNAME.COLUMN-2)")


# In[ ]:


# PRINT NUNIQUE OF ALL COLUMNS

def nunique_column_all(df):
    for col in df.columns:
        print(df[col].value_counts())
        print()


# In[4]:


# PRINT NUNIQUE OF COLUMNS THAT ARE OBJECTS

def nunique_column_objects(df): 
    for col in df.columns:
        if df[col].dtypes == 'object':
            print(f'{col} has {df[col].nunique()} unique values.')


# In[5]:


# PRINT NUNIQUE OF COLUMNS THAT ARE *NOT* OBJECTS

def nunique_column_qty(df): 
    for col in df.columns:
        if df[col].dtypes != 'object':
            print(f'{col} has {df[col].nunique()} unique values.')


# In[ ]:


# **COMPUTE RANGE FOR ALL NUMERIC VARIABLES**

def numeric_range(df):
    numeric_list = df.select_dtypes(include = 'float').columns.tolist()
    numeric_range = df[numeric_list].describe().T
    numeric_range['range'] = numeric_range['max'] - numeric_range['min']
    return numeric_range


# In[ ]:


# BUILD A FUNCTION THAT DOES THIS FOR ALL "FLOAT" COLUMNS

# float_cols = train_iris.select_dtypes(include='float').columns

# Plot numeric columns
#plot_float_cols = float_cols 
#for col in plot_float_cols:
#    plt.hist(train_iris[col])
#    plt.title(col)
#    plt.show()
#    plt.boxplot(train_iris[col])
#    plt.title(col)
#    plt.show()


# In[ ]:


# BUILD A FUNCTION THAT DOES THIS FOR ALL "OBJECT" COLUMNS

# train.species.value_counts()
# plt.hist(train_iris.species_name)


# In[ ]:


# BUILD A FUNCTION THAT DOES THIS

#test_var = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
#for var in test_var:
#    t_stat, p_val = t_stat, p_val = stats.mannwhitneyu(virginica[var], versicolor[var], alternative="two-sided")
#    print(f'Comparing {var} between Virginica and Versicolor')
#    print(t_stat, p_val)
#    print('')
#    print('---------------------------------------------------------------------')
#    print('')


# In[ ]:


# sns.pairplot(DF, hue='TARGET_COLUMN', corner=True)
# plt.show()

