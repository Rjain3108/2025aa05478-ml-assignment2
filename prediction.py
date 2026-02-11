import joblib
import pandas as pd
from utils.data_import import read_csv,get_feature_target_data

logisticModel = joblib.load("pkl/logisticModel.pkl")


model_functions = {
    "Logistic Regression": logisticModel
}
data= read_csv("test.csv")
X_test, y_test = get_feature_target_data(data)

def predict(modelName, X_test):
    model = model_functions[modelName]
    y_pred = model.predict(X_test)
    print(f"Predicted values using {modelName}: {y_pred}")
    return y_pred

def main():
    modelName = "Logistic Regression"
    predict(modelName, X_test)
    
if __name__ == "__main__":
    main()