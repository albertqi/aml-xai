# Explain the model using SHAP.
# https://arxiv.org/abs/1705.07874


import shap
from model import train


def explain():
    # Get the SHAP values for the model.
    X_train, model = train()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_train)

    # Plot the SHAP values.
    i = 43
    shap.plots.beeswarm(shap_values)
    shap.plots.waterfall(shap_values[i])


if __name__ == "__main__":
    explain()
