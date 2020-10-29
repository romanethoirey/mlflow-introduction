import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle


def preprocessing(app_train, app_test, app_train_balanced):

    targets = app_train

    # Replace Own car by binary term
    app_train['FLAG_OWN_CAR'].replace(to_replace=['N'], value=0, inplace=True)
    app_train['FLAG_OWN_CAR'].replace(to_replace=['Y'], value=1, inplace=True)

    app_test['FLAG_OWN_CAR'].replace(to_replace=['N'], value=0, inplace=True)
    app_test['FLAG_OWN_CAR'].replace(to_replace=['Y'], value=1, inplace=True)

    app_train_balanced['FLAG_OWN_CAR'].replace(to_replace=['N'], value=0, inplace=True)
    app_train_balanced['FLAG_OWN_CAR'].replace(to_replace=['Y'], value=1, inplace=True)

    # Replace Own realty by binary term
    app_train['FLAG_OWN_REALTY'].replace(to_replace=['N'], value=0, inplace=True)
    app_train['FLAG_OWN_REALTY'].replace(to_replace=['Y'], value=1, inplace=True)

    app_test['FLAG_OWN_REALTY'].replace(to_replace=['N'], value=0, inplace=True)
    app_test['FLAG_OWN_REALTY'].replace(to_replace=['Y'], value=1, inplace=True)

    app_train_balanced['FLAG_OWN_REALTY'].replace(to_replace=['N'], value=0, inplace=True)
    app_train_balanced['FLAG_OWN_REALTY'].replace(to_replace=['Y'], value=1, inplace=True)

    # Replace NAME_CONTRACT_TYPE by binary values
    app_train['NAME_CONTRACT_TYPE'].replace(to_replace=['Cash loans'], value=1, inplace=True)
    app_train['NAME_CONTRACT_TYPE'].replace(to_replace=['Revolving loans'], value=2, inplace=True)

    app_test['NAME_CONTRACT_TYPE'].replace(to_replace=['Cash loans'], value=1, inplace=True)
    app_test['NAME_CONTRACT_TYPE'].replace(to_replace=['Revolving loans'], value=2, inplace=True)

    app_train_balanced['NAME_CONTRACT_TYPE'].replace(to_replace=['Cash loans'], value=1, inplace=True)
    app_train_balanced['NAME_CONTRACT_TYPE'].replace(to_replace=['Revolving loans'], value=2, inplace=True)

    app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243

    app_train['DAYS_EMPLOYED'] = app_train['DAYS_EMPLOYED'].replace({365243: np.nan})

    app_test['DAYS_EMPLOYED_ANOM'] = app_test["DAYS_EMPLOYED"] == 365243

    app_test['DAYS_EMPLOYED'] = app_test['DAYS_EMPLOYED'].replace({365243: np.nan})

    app_train_balanced['DAYS_EMPLOYED_ANOM'] = app_train_balanced["DAYS_EMPLOYED"] == 365243

    app_train_balanced['DAYS_EMPLOYED'] = app_train_balanced['DAYS_EMPLOYED'].replace({365243: np.nan})

    app_train = pd.get_dummies(app_train, drop_first=True)
    app_test = pd.get_dummies(app_test, drop_first=True)
    app_train_balanced = pd.get_dummies(app_train_balanced, drop_first=True)

    most_corr_features = ["DEF_60_CNT_SOCIAL_CIRCLE",
                          "DEF_30_CNT_SOCIAL_CIRCLE",
                          "LIVE_CITY_NOT_WORK_CITY",
                          "OWN_CAR_AGE",
                          "DAYS_REGISTRATION",
                          "OCCUPATION_TYPE_Laborers",
                          "FLAG_DOCUMENT_3",
                          "REG_CITY_NOT_LIVE_CITY",
                          "FLAG_EMP_PHONE",
                          "REG_CITY_NOT_WORK_CITY",
                          "DAYS_ID_PUBLISH",
                          "CODE_GENDER_M",
                          "DAYS_LAST_PHONE_CHANGE",
                          "NAME_INCOME_TYPE_Working",
                          "REGION_RATING_CLIENT",
                          "REGION_RATING_CLIENT_W_CITY",
                          "DAYS_EMPLOYED",
                          "DAYS_BIRTH",
                          "EXT_SOURCE_3",
                          "EXT_SOURCE_2",
                          "EXT_SOURCE_1",
                          "NAME_INCOME_TYPE_Pensioner",
                          "DAYS_EMPLOYED_ANOM",
                          "ORGANIZATION_TYPE_XNA",
                          "FLOORSMAX_AVG",
                          "FLOORSMAX_MEDI",
                          "FLOORSMAX_MODE",
                          "AMT_GOODS_PRICE",
                          "REGION_POPULATION_RELATIVE",
                          "ELEVATORS_AVG",
                          "ELEVATORS_MEDI",
                          "FLOORSMIN_AVG",
                          "FLOORSMIN_MEDI",
                          "WALLSMATERIAL_MODE_Panel",
                          "LIVINGAREA_AVG",
                          "LIVINGAREA_MEDI",
                          "FLOORSMIN_MODE"
                          ]

    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    app_train[most_corr_features] = imputer.fit_transform(app_train[most_corr_features])
    app_train_balanced[most_corr_features] = imputer.fit_transform(app_train_balanced[most_corr_features])

    app_train = app_train[most_corr_features]
    app_test = app_test[most_corr_features]
    app_train_balanced = app_train_balanced[most_corr_features]

    scaler = MinMaxScaler(feature_range=(0, 1))
    imputer.fit(app_train)
    app_train = imputer.transform(app_train)
    app_test = imputer.transform(app_test)
    app_train_balanced = imputer.transform(app_train_balanced)

    app_train = scaler.fit_transform(app_train)
    app_test = scaler.fit_transform(app_test)
    app_train_balanced = scaler.fit_transform(app_train_balanced)

    return app_train, app_test, app_train_balanced
