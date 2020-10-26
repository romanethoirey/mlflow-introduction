# numpy and pandas for data manipulation
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


def xgboost_func(app_train):
    X = app_train.drop(columns=['TARGET'])
    y = app_train["TARGET"]

    print("XGBoost : Splitting data")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    print("XGBoost : Fitting data")
    model_xgb = XGBClassifier()
    model_xgb.fit(X_train, y_train)

    print("XGBoost : Predict data")
    predictions = model_xgb.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    f1_score_xgb = f1_score(y_test, predictions)

    return model_xgb, predictions, accuracy, rmse, f1_score_xgb
