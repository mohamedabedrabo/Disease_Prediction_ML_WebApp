#!/usr/bin/env python
# coding: utf-8

# ## Heart Disease Predictive System
# ### Using Logistic Regression Machine Learning Algorithm

# In[1]:


# Why logistic regression could be good fit for this use case ?
# Because it's a binary classification case.


# ## Import Libraries

# In[2]:


# For data analysis and math:
import numpy as np
import pandas as pd
# To split to train and test set:
from sklearn.model_selection import train_test_split
# The model logistic regrission:
from sklearn.linear_model import LogisticRegression
# To calculate accuracy:
from sklearn.metrics import accuracy_score


# ## 1. Data Collection And Analysis

# #### Load data:

# In[3]:


heart_data = pd.read_csv(r'C:\Python\5_Disease_Prediction_App\Multiple Disease Prediction System\heart_disease_data.csv')


# #### First Look:

# In[4]:


# Last column: target, other columns: features
heart_data.head()


# In[5]:


heart_data.tail()


# In[6]:


# We have 303 rows, 14 columns
heart_data.shape


# In[7]:


heart_data.info()


# #### Check null values:

# In[8]:


# We dont have any missing values
heart_data.isnull().sum()


# #### Statistical information:

# In[9]:


heart_data.describe()


# In[10]:


# 0 doesn't have disease, 165 have
# So we have a good distrebution between 2 of them
heart_data.target.value_counts()


# ## 2. Data Pre-processing And Standrization

# #### Split features and target:

# In[12]:


X = heart_data.drop(columns = 'target', axis = 1)
Y = heart_data['target']


# In[13]:


print(X)


# In[14]:


print(Y)


# ## 3. Split To Test And Train

# In[15]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,
                                                   stratify = Y,
                                                   random_state=2)


# In[18]:


print(X.shape, X_train.shape, X_test.shape)


# ## 4. Run And Check The Model:

# In[19]:


model = LogisticRegression()


# In[22]:


model.fit(X_train, Y_train)


# #### Accuracy on training set:

# In[23]:


X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[24]:


# Accuracy
print(training_data_accuracy)


# #### Accuracy on test set:

# In[25]:


X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[26]:


print(test_data_accuracy)


# ## 5. Build Predictive System on new data:

# In[27]:


def prediction_fun2 (input_data) :
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = model.predict(input_data_reshaped)
    if (prediction[0] == 0):
        return('The person is not ill')
    else :
        return('The person is ill')


# In[28]:


input_data = (41,0,1,130,204,0,0,172,0,1.4,2,0,2)
prediction_fun2(input_data)


# In[29]:


input_data2 = (67,1,0,160,286,0,0,108,1,1.5,1,3,2)
prediction_fun2(input_data2)


# ## 6. Save the model : 

# In[30]:


import pickle


# In[31]:


filename = 'heart_model.sav'
pickle.dump(model, open(filename, 'wb'))

