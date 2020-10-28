import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

def randomforest_func(app_train, labels):
    X = app_train
    y = labels

    print("Random Forest : Splitting data")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    print("Random Forest : Fitting data")
    model_randomforest = RandomForestClassifier(max_depth=2, random_state=0)
    model_randomforest.fit(X_train, y_train)

    print("Random Forest : Predict data")
    predictions = model_randomforest.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    f1_score_model = f1_score(y_test, predictions)

    return model_randomforest, predictions, accuracy, rmse, f1_score_model