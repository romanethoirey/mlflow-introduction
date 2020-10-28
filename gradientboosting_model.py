import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

def gradientboosting_func(app_train, labels):
    X = app_train
    y = labels
    lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]
    list_models = []
    list_accur = []

    print("Gradient Boosting : Splitting data")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    print("Gradient Boosting : Fitting data")
    for learning_rate in lr_list:
        model_gdb = GradientBoostingClassifier(n_estimators=20, learning_rate=learning_rate, max_features=2, max_depth=2,
                                            random_state=0)
        model_gdb.fit(X_train, y_train)

        predictions = model_gdb.predict(X_test)
        list_accur.append(accuracy_score(y_test, predictions))
        list_models.append(model_gdb)

    print("Gradient Boosting : Predict data")
    accuracy = max(list_accur)
    index = list_accur.index(max(list_accur))
    model_gdb = list_models[index]
    predictions = model_gdb.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    f1_score_model = f1_score(y_test, predictions)

    return model_gdb, predictions, accuracy, rmse, f1_score_model