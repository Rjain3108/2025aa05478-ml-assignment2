import streamlit as st
import pandas as pd
import utils.data_import as data_import
import prediction as pred
from utils.evaluation_metrix import evaluate_model

test_file_path = "test.csv"
X_test,y_test = None, None
st.title("Welcome to Mobile pricing prediction model !!!")
st.text("Download the test data to predict the price range of mobile phones")
with open(test_file_path, "rb") as f:
    st.download_button(
        "Download CSV",
        data=f,
        file_name="test.csv",
        mime="text/csv"
    )
st.text("Select the model using that you want to predict some result")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
if uploaded_file is not None:
    X_test,y_test = data_import.get_feature_target_data(pd.read_csv(uploaded_file))

else:
    st.write("Please upload a CSV file to proceed.")
#test_data = data_import.get_mobile_test_data()
# Create a dropdown menu for selecting a hobby
model = st.selectbox("Select a model:", ['ALL','Logistic Regression', 'Decision Tree', 'kNN', 'Naive Bayes', 'Random Forest', 'XGBoost'])

# A button that displays text when clicked
if st.button("Calculate"):
    if X_test is not None and y_test is not None:
        y_pred = pred.predict(model, X_test)
        st.write(f"Accuracy: {evaluate_model(y_test, y_pred)}")
        st.write(f"Predicted values: {y_pred}")
    else:
        st.write("Error: Please upload a CSV file to proceed.")