#!/usr/bin/env python
# coding: utf-8

# ## Diabetes Predictive System
# ### Using Support Vector Machines Machine Learning Algorithm

# ## Import The Libraries

# In[3]:


# For data analysis and math:
import numpy as np
import pandas as pd
# To standrize the data:
from sklearn .preprocessing import StandardScaler
# To split to train and test set:
from sklearn.model_selection import train_test_split
# The model support vector machines:
from sklearn import svm
# To calculate accuracy:
from sklearn.metrics import accuracy_score


# ## 1. Data Collection And Analysis

# #### Load the dataset to pandas df:

# In[4]:


diabetes_dataset = pd.read_csv(r'diabetes.csv')


# #### First look at the data:

# In[5]:


# Features : first 8 columns, information about the patient
# Label : last column, 0 not affected, 1 affected
diabetes_dataset.head()


# #### Num of rows and columns:

# In[6]:


# (rows, columns), 8 features, 1 Label
diabetes_dataset.shape


# #### Statistical information:

# In[7]:


# For Glucose 25% of the values are less than 62.
diabetes_dataset.describe()


# #### how many have diabetes and how many dont:

# In[8]:


# 0: Non-Diabetic, 1:Diabetic
diabetes_dataset.Outcome.value_counts()


# #### Mean values for all features, grouped by label:

# In[9]:


# It's obvious that Diabetic people have higher values in all the tests
diabetes_dataset.groupby('Outcome').mean()


# ## 2. Data Pre-processing And Standrization

# #### Split features and labels:

# In[10]:


# Features:X, Labels:Y
X = diabetes_dataset.drop(columns = 'Outcome', axis = 1)
Y = diabetes_dataset['Outcome']


# In[11]:


X.head()


# In[12]:


Y.head()


# #### Data standrization:

# In[13]:


# We have different range of values in each column and this model cant handle this
# So we need to standrize the value for the model to work properly


# In[14]:


scaler = StandardScaler()


# In[15]:


scaler.fit(X)


# In[16]:


standrdized_data = scaler.transform(X)


# In[17]:


print(standrdized_data)


# In[18]:


# Call it X again :
X = standrdized_data


# ## 3. Split To Train And Test Set

# #### Split:

# In[19]:


# stratify : make proper split based on the label values, if we dont do this
# we might have test set full of 0 values or only 1 values
# random state : split type
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,
                                                    stratify = Y, random_state = 2)


# #### Check shapes:

# In[20]:


print(X.shape, X_train.shape, X_test.shape)


# 
# ## 4. Run And Test The Model - SVM Model

# #### Creating the model:

# In[21]:


# using SVM on linear model
classifier = svm.SVC(kernel = 'linear')


# #### Train the model:

# In[22]:


classifier.fit(X_train, Y_train)


# #### Check accuracy - training data:

# In[23]:


# Accuracy score on training data
# Comparing real labels vs. the values the model will predict


# In[24]:


# Predicted labels on training set:
X_train_prediction = classifier.predict(X_train)


# In[25]:


# Check accuracy, we pass 2 inputs
# (the predicted labels of training set, the real labels of training set)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[26]:


# It's not a bad score because we have very small dataset
print('Accuracy score of trianing data',training_data_accuracy)


# #### Check accuracy - test data:

# In[27]:


X_test_prediction = classifier.predict(X_test)


# In[28]:


test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[29]:


# It's a good score compare to the size of the dataset
print('Accuracy score of trianing data',test_data_accuracy)


# ## 5. Create Predictive System For New Data

# #### Input data:

# In[30]:


# we know this case is non-diabetec so it should predict 0
input_data = (4,110,92,0,0,37.6,0.191,30)


# In[31]:


# Convert to NumPy array:
input_data_as_numpy_array = np.asarray(input_data)


# In[32]:


# Reshape the array as we are expecting only 1 datapoint
# the model will expect 768 data point, this will tell it 
# that we are predicting only one data point
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


# #### Standrize input data:

# In[33]:


std_data = scaler.transform(input_data_reshaped)


# In[34]:


print(std_data)


# #### Predict:

# In[35]:


prediction = classifier.predict(std_data)


# In[36]:


# so it predicted that it's 0 (non-diabetec)
print(prediction)


# #### Build a function that do all the job of prediction:

# In[37]:


def prediction_fun (input_data) :
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    std_data = scaler.transform(input_data_reshaped)
    prediction = classifier.predict(std_data)
    if (prediction[0] == 0):
        return('The person is not diabetec')
    else :
        return('The person is diabetec')


# In[38]:


prediction_fun(input_data)


# ## 6. Save the model:

# In[39]:


import pickle


# In[40]:


filename = 'diabetes_model.sav'
pickle.dump(classifier, open(filename, 'wb'))

