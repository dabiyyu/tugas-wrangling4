import pandas as pd
import pickle

from helper.data_check_preparation import read_and_check_data
from helper.feature_engineering import feature_engineering
from helper.constant import TRAIN_COLUMN, PATH
from helper.models import LINEAR_MODEL_CLF

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

def train_model():
    # pembacaan dan pengecekan data
    cars_data = read_and_check_data(PATH, TRAIN_COLUMN)
    
    # feature engineering
    X_train_transformed = feature_engineering(cars_data = cars_data, state = "train")
    print("Start Saving Result Feature Engineering in Train Data !")
    X_train_transformed.to_csv("artifacts/X_train_transformed.csv")
    
    # feature engineering
    X_test_transformed = feature_engineering(cars_data =cars_data, state = "test")
    print("Start Saving Result Feature Engineering in Test Data!")
    X_test_transformed.to_csv("artifacts/X_test_transformed.csv")

    # siapkan fitur data dan target data
    X = cars_data.drop(["price"], axis=1)
    y = cars_data["price"]

    # data splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

    # define model and start training
    clf_model = LINEAR_MODEL_CLF["logreg_cv"]
    clf_model.fit(X_train, y_train)
    y_pred_class = clf_model.predict_proba(X_test).argmax(1)
    y_pred_proba = clf_model.predict_proba(X_test)[:, 1]

    pickle.dump(clf_model, open("artifacts/logreg_model.pkl", "wb"))
    
    # show training result
    print("------------------------------")
    print("Model Performance:")
    print("ROC_AUC:", roc_auc_score(y_test, y_pred_proba))
    print("Recall:", recall_score(y_test, y_pred_class))
    print("Precision:", precision_score(y_test, y_pred_class))
    print("f1_score:", f1_score(y_test, y_pred_class, average="macro"))

if __name__ == "__main__":
    print("START RUNNING PIPELINE!")
    train_model()
    print("FINISH RUNNING PIPELINE!")
