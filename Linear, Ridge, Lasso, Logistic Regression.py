#!/usr/bin/env python
# coding: utf-8

# ## Linear Regression, Ridge and Lasso

# In[1]:


#house pricing dataset
from sklearn.datasets import load_boston


# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[14]:


df = load_boston()
df


# In[13]:


# converting df into a DataFrame
dataset = pd.DataFrame(df.data)
dataset.columns = df.feature_names
dataset.head()
# these features (i.e CRIM, ZN,..., LSTAT) all are independent features


# In[15]:


# Now, we will create the dependent feature (i.e Price)
dataset["Price"] = df.target


# In[16]:


dataset.head()


# In[17]:


# Dividing the dataset into independent and dependent features
# X = independent freatures: column-0 to column-12
# Y = dependent freature: column-13
X = dataset.iloc[:,:-1]  # from all the columns skip the last column(i.e price)
Y = dataset.iloc[:,-1] # take the last column


# In[18]:


X.head()


# In[19]:


Y.head()


# In[43]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
# test_size = 0.33 means 33% data is used to test and the rest 77% to train the model


# In[44]:


# Now, we will perform Linear Regression 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score  # cross validation makes different combinations of test data and training data

lin_reg = LinearRegression()  # import LinearRegression as lin_reg
mse = cross_val_score(lin_reg, X_train, Y_train, scoring='neg_mean_squared_error', cv=5)
print(mse)
mean_mse = np.mean(mse)
print(mean_mse)


# In[60]:


# Fit the LinearRegression model
lin_reg.fit(X_train, Y_train)  # Training the model using independent (X) and dependent (Y) features

# Using the predict() function of the LinearRegression model
y_pred = lin_reg.predict(X_test)
print(y_pred)


# In[64]:


from sklearn.metrics import r2_score
r2_score_linearReg = r2_score(Y_test, y_pred)  # Y_test are the read values and y_pred are the predicted values from linear regression
print(r2_score_linearReg)
# the value of r2_score_linearReg must be close to 1 (i.e 100%) for higher accuracy


# In[47]:


# Cross-Validation (mse and mean_mse): Helps you understand how well your linear regression model is likely to perform on unseen data and 
#detect potential overfitting or underfitting.For example, a lower mean MSE indicates better performance.

#The predict() function is used after the model is trained (using .fit()) to make predictions on new data.


# In[51]:


# Ridge Regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV # used for hyper-parameter tunning

ridge = Ridge()
params = {'alpha':[1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 100]}
# we are taking so many alpha values because GridSearchCV will check all the values and give us the best value(i.e the value when model works the best)
ridge_regressor = GridSearchCV(ridge, params, scoring='neg_mean_squared_error', cv=10)
ridge_regressor.fit(X_train,Y_train)


# In[52]:


print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)


# In[55]:


# -25.18 => Linear Regression & -25.47 => Ridge Regression
# When comparing NMSE values, a decrease means that the MSE is increasing
# Therefore, going from -25.18 to -25.47 means the MSE has increased from 25.18 to 25.47. 
# This indicates that the Ridge Regression model's predictions are bad than the Linear Regression model's predictions.
# Note: Our target is to go as close to 0 as possible

# Lasso Regression
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV # used for hyper-parameter tunning

lasso = Lasso()
params = {'alpha':[1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 100]}
# we are taking so many alpha values because GridSearchCV will check all the values and give us the best value(i.e the value when model works the best)
lasso_regressor = GridSearchCV(lasso, params, scoring='neg_mean_squared_error', cv=10)
lasso_regressor.fit(X_train,Y_train)


# In[56]:


print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)


# In[62]:


# using predict() for Ridge
Y_pred = ridge_regressor.predict(X_test)
print(Y_pred)


# In[63]:


from sklearn.metrics import r2_score
r2_score_ridge = r2_score(Y_test, Y_pred)  # Y_test are the real values and Y_pred are the predicted values of Ridge Regression
print(r2_score_ridge)
# the value of r2_score_ridge must be close to 1 (i.e 100%) for higher accuracy


# In[ ]:




