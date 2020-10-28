# This is a sample Python script.
import sys

import mlflow
import pandas as pd

from xgboost_model import xgboost_func
from preprocessing import preprocessing
from shap_model import shap_implementation
from randomforest_model import randomforest_func
from gradientboosting_model import gradientboosting_func

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app_train = pd.read_csv("data/application_train.csv")
    app_test = pd.read_csv("data/application_test.csv")

    print("Preprocessing Data")

    labels = app_train["TARGET"]
    app_train, app_test, app_train_balanced = preprocessing(app_train, app_test)

    try:
        if sys.argv[1] == "XGBoost":
            print("______________")
            print("Xgboost")
            model, pred, accur, rmse, f1_score = xgboost_func(app_train, labels)
            #model_balanced, pred_balanced, accur_balanced, rmse_balanced, f1_score_balanced = xgboost_func(app_train_balanced, labels)

        elif sys.argv[1] == "Gradient Boosting":
            print("______________")
            print("Gradient Boosting")
            model, pred, accur, rmse, f1_score = gradientboosting_func(app_train, labels)
            # model_balanced, pred_balanced, accur_balanced, rmse_balanced, f1_score_balanced = gradientboosting_func(app_train_balanced, labels)

        elif sys.argv[1] == "Random Forest":
            print("______________")
            print("Random Forest")
            model, pred, accur, rmse, f1_score = randomforest_func(app_train, labels)
            # model_balanced, pred_balanced, accur_balanced, rmse_balanced, f1_score_balanced = randomforest_func(app_train_balanced, labels)

        print("______________")
        print(sys.argv[1])
        print("The accuracy of the model is {}".format(accur))
        print("The RMSE of the model is {}".format(rmse))
        print("The F1 Score of the model is {}".format(f1_score))

        # print("The accuracy of the balanced model is {}".format(accur_balanced))
        # print("The RMSE of the balanced model is {}".format(rmse_balanced))
        # print("The F1 Score of the balanced model is {}".format(f1_score_balanced))

    except IndexError:
        logger.exception(
            "Please provide a model to execute among: XGBoost, Gradient Boost or Random Forest"
        )
        raise

    with mlflow.start_run():

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("Accuracy", accur)
        mlflow.log_metric("f1_score", f1_score)

        # mlflow.log_metric("balanced rmse", rmse_balanced)
        # mlflow.log_metric("balanced Accuracy", accur_balanced)
        # mlflow.log_metric("balanced f1_score", f1_score_balanced)

        mlflow.sklearn.log_model(model, sys.argv[1])
        # mlflow.sklearn.log_model(model_balanced, sys.argv[1]+" balanced")

    shap_implementation(model, app_train)
