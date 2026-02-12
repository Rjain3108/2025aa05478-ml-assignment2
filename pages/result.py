import streamlit as st
import pandas as pd
import numpy as np
import json
from utils.evaluation_metrix import plot_confusion_matrix, plot_evaluation_metrics

file_path = "model_results.json"
st.title("Mobile Pricing – Result Metrics Dashboard")

model = ['Logistic Regression','Decision Tree', 'kNN']

try:
    with open(file_path, "r") as f:
        model_results_data = json.load(f)
except FileNotFoundError:
            st.write(f"matrix data not found.")

st.divider()
st.markdown("<h3 style='text-align:center;'>Confusion Matrix</h3>", unsafe_allow_html=True)
cols = st.columns(len(model))
for col, m in zip(cols, model):
    with col:
        st.text(f"{m}")
        try: 
            fig = plot_confusion_matrix(m, np.array(model_results_data[m]["confusion_matrix"]))
            st.pyplot(fig)
            st.dataframe(model_results_data[m]["multilabel_confusion_matrix"], use_container_width=True)
        except KeyError:
            st.write(f"Multilabel confusion matrix/ confusion matrix data for {m} not found.")
        continue

st.divider()
st.markdown("<h3 style='text-align:center;'>Evaluation Metrics</h3>", unsafe_allow_html=True)
cols = st.columns(len(model)) 
for col, m in zip(cols, model):
    with col:
        try: 
            metrics = model_results_data[m]["evaluation_metrics"]
            fig = plot_evaluation_metrics(m, metrics)
            st.pyplot(fig)
        except KeyError:
            st.write(f"Evaluation metrics data for {m} not found.")
        continue

table_data = {}
for model_name, model_data in model_results_data.items():
    table_data[model_name] = model_data["evaluation_metrics"]

df_metrics = pd.DataFrame.from_dict(table_data, orient="index").round(3)
st.subheader("Evaluation Metrics Table")
st.dataframe(df_metrics, use_container_width=True)