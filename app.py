import streamlit as st
st.title("Welcome to Prediction model for problem !!!")
st.text("Select the model using that you want to predict some result")


# Create a dropdown menu for selecting a hobby
model = st.selectbox("Select a model:", ['Logistics Regression', 'Decision Tree', 'kNN', 'Naive Bayes', 'Random Forest', 'XGBoost'])
dataType = st.selectbox("Select a dataType:", ['file','Manually data enter'])


# A button that displays text when clicked
if st.button("Calculate"):
    st.write("Predicting value using model " + model + '.')