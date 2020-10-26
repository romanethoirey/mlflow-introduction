# This is a sample Python script.
import sys

import mlflow
import pandas as pd

from xgboost_model import xgboost_func
from preprocessing import preprocessing
from shap_model import shap_implementation

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app_train = pd.read_csv("data/application_train.csv")
    app_test = pd.read_csv("data/application_test.csv")

    print("Preprocessing Data")
    app_train, app_test = preprocessing(app_train, app_test)

    if sys.argv[1] == "XGBoost":
        print("Xgboost")
        model, pred, accur, rmse, f1_score = xgboost_func(app_train)

    elif sys.argv[1] == "Gradient Boost":
        breakpoint
    else:
        logger.exception(
            "Please provide a model to execute among: XGBoost, Gradient Boost or ..."
        )

    with mlflow.start_run():

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("Accuracy", accur)
        mlflow.log_metric("f1_score", f1_score)

        mlflow.sklearn.log_model(model, sys.argv[1])

    shap_implementation(model, app_train)