import streamlit as st
import pandas as pd

import test_data_import
st.title("Welcome to Prediction model for problem !!!")
st.text("Select the model using that you want to predict some result")

test_data = test_data_import.get_mobile_price_test()
# Create a dropdown menu for selecting a hobby
model = st.selectbox("Select a model:", ['Logistic Regression', 'Decision Tree', 'kNN', 'Naive Bayes', 'Random Forest', 'XGBoost'])
dataType = st.selectbox("Select a dataType:", ['file','Manually data enter'])

model2 = st.selectbox("Select a model2:", ['Logistic Regression', 'Decision Tree', 'kNN', 'Naive Bayes', 'Random Forest', 'XGBoost'])


# A button that displays text when clicked
if st.button("Calculate"):
    st.write(f"Predicting value using model {model} and {model2}")
    st.dataframe(test_data.head())
