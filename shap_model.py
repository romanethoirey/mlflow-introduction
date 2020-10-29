import shap


def shap_implementation(model, app_train):
    shap.initjs()
    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(app_train)
    shap.force_plot(explainer.expected_value, shap_values[2], app_train[2])
    shap.summary_plot(shap_values, app_train)
