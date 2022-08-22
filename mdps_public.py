# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 12:48:13 2022

@author: MohamedAbedrabo
"""

# 1: Import libraries:
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd

# Import prediction functions from the models we built:
from Diabetes_Prediction_SVM import prediction_fun
from Heart_Disease_Prediction_LR import prediction_fun2

# 2: Loading the saved models:
diabetes_model = pickle.load(open('diabetes_model.sav','rb'))


heart_disease_model = pickle.load(open('heart_model.sav','rb'))


# 3: Sidebar for navigation:
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction',
                            'Heart Disease Prediction'],
                           
                           icons = ['activity','heart' ],
                           
                           default_index = 0)
# default index=0 says that the app will open on Diabetes Prediction page
# icons names : https://icons.getbootstrap.com/
# the icons will be displayed for each element on sidebar by the order


# 4: Diabates prediction page:
if (selected == 'Diabetes Prediction'):
    
    # Page title:
    st.title('Diabates prediction using ML')
    
    # Getting input data from the user :
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number Of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure')
    with col1:
        SkinThickness = st.text_input('Skin Thickness')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function')
    with col2:
        Age = st.text_input('Age')
    
    # Prediction Value:
    diab_diagnosis = ''
    
    # Prediction Button:
    if st.button('Diabetes Test Result'):
        diab_prediction = prediction_fun([Pregnancies, Glucose,BloodPressure, SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
        # Prediction Functionality:
        diab_diagnosis = diab_prediction

    st.success(diab_diagnosis)



# 5: Heart prediction page:
if (selected == 'Heart Disease Prediction'):
    
    # Page title:
    st.title('Heart Disease prediction using ML')
    
    # Getting input data from the user :
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.text_input('Sex')
    with col3:
        cp = st.text_input('CP')
    with col1:
        trestbps = st.text_input('Trestbps')
    with col2:
        chol = st.text_input('Chol')
    with col3:
        fbs = st.text_input('Fbs')
    with col1:
        restecg = st.text_input('Restecg')
    with col2:
        thalach = st.text_input('Thalach')
    with col3:
        exang = st.text_input('Exang')
    with col1:
        oldpeak = st.text_input('Oldpeak')
    with col2:
        slope = st.text_input('Slope')
    with col3:
        ca = st.text_input('CA')
    with col1:
        thal = st.text_input('Thal')
    
    # Prediction Value:
    heart_diagnosis = ''
    
    # Prediction Button:
    if st.button('Heart Disease Test Result'):
        
        heart_input = np.array([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal], dtype=float)
        heart_prediction = prediction_fun2([heart_input])

        #Prediction Functionality:
        heart_diagnosis = heart_prediction


    st.success(heart_diagnosis)
    
    



    
    