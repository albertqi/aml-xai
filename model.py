# Train an XGBoost model on the BAF dataset.
# https://arxiv.org/abs/2211.13358


import matplotlib.pyplot as plt
import xgboost as xgb
from eda import preprocess
from sklearn.metrics import roc_auc_score, roc_curve


TARGET_FPR = 0.05


def train():
    # Preprocess the data.
    df = preprocess()

    # Split data into features and target.
    X = df.drop(columns=["fraud_bool"], axis=1)
    y = df["fraud_bool"]

    # Split data into training and testing sets.
    X_train, X_test = X[X["month"] < 6], X[X["month"] >= 6]
    y_train, y_test = y[X["month"] < 6], y[X["month"] >= 6]

    # Train the model.
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)

    # Evaluate the model.
    y_pred = model.predict_proba(X_test)[:, 1]
    fprs, tprs, _ = roc_curve(y_test, y_pred)
    tpr = tprs[fprs <= TARGET_FPR][-1]
    fpr = fprs[fprs <= TARGET_FPR][-1]
    auc = roc_auc_score(y_test, y_pred)

    # Print the results.
    print(f"TPR: {tpr:.2f}")
    print(f"FPR: {fpr:.2f}")
    print(f"AUC: {auc:.2f}")

    # Plot the ROC curve.
    plt.plot(fprs, tprs)
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve")
    plt.show()

    # Return the training data and model.
    return X_train, model


if __name__ == "__main__":
    train()
