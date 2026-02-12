import joblib
import pandas as pd
from utils.data_import import read_csv,get_feature_target_data


model_functions = {
    "Logistic Regression": joblib.load("pkl/logisticModel.pkl")
    #"Decision Tree": joblib.load("pkl/decisionTreeModel.pkl"),
    #"kNN": joblib.load("pkl/kNNModel.pkl"), 
    #"Naive Bayes": joblib.load("pkl/naiveBayesModel.pkl"), 
    #"Random Forest": joblib.load("pkl/randomForestModel.pkl"), 
    #"XGBoost": joblib.load("pkl/xgboostModel.pkl")
}
data= read_csv("test.csv")
X_test, y_test = get_feature_target_data(data)

def predict(modelName, X_test):
    model = model_functions[modelName]
    y_pred = model.predict(X_test)
    return y_pred

def main():
    modelName = "Logistic Regression"
    predict(modelName, X_test)
    
if __name__ == "__main__":
    main()