# Explain the model using SHAP.
# https://arxiv.org/abs/1705.07874


import shap
from model import train


def explain():
    # Get the SHAP values for the model.
    X_test, model = train()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    # Plot the SHAP values.
    i = 21
    shap.plots.beeswarm(shap_values)
    shap.plots.waterfall(shap_values[i])

    # Return the testing data, model, and explainer.
    return X_test, model, explainer


if __name__ == "__main__":
    explain()
