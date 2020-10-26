import pandas as pd

def preprocessing(app_train, app_test):
    # Replace Own car by binary term
    app_train['FLAG_OWN_CAR'].replace(to_replace=['N'], value=0, inplace=True)
    app_train['FLAG_OWN_CAR'].replace(to_replace=['Y'], value=1, inplace=True)

    app_test['FLAG_OWN_CAR'].replace(to_replace=['N'], value=0, inplace=True)
    app_test['FLAG_OWN_CAR'].replace(to_replace=['Y'], value=1, inplace=True)

    # Replace Own realty by binary term
    app_train['FLAG_OWN_REALTY'].replace(to_replace=['N'], value=0, inplace=True)
    app_train['FLAG_OWN_REALTY'].replace(to_replace=['Y'], value=1, inplace=True)

    app_test['FLAG_OWN_REALTY'].replace(to_replace=['N'], value=0, inplace=True)
    app_test['FLAG_OWN_REALTY'].replace(to_replace=['Y'], value=1, inplace=True)

    # Replace NAME_CONTRACT_TYPE by binary values
    app_train['NAME_CONTRACT_TYPE'].replace(to_replace=['Cash loans'], value=1, inplace=True)
    app_train['NAME_CONTRACT_TYPE'].replace(to_replace=['Revolving loans'], value=2, inplace=True)

    app_test['NAME_CONTRACT_TYPE'].replace(to_replace=['Cash loans'], value=1, inplace=True)
    app_test['NAME_CONTRACT_TYPE'].replace(to_replace=['Revolving loans'], value=2, inplace=True)

    app_train = pd.get_dummies(app_train)
    app_test = pd.get_dummies(app_test)

    return app_train, app_test
