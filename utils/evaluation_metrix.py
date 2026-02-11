from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import json


def evaluate_model(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    cm_list = cm.tolist()
    data_to_save = {
    "confusion_matrix": cm_list
}
    with open("confusion_matrix.json", "w") as f:
        json.dump(data_to_save, f, indent=4)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return accuracy_score(y_test, y_pred)