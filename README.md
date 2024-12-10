# Assessing Properties of Explanations for Anti-Money Laundering Machine Learning Models with Domain Experts

## Albert Qi and Steve Dalla

### Fall 2024

When a bank receives a transaction, how can they be certain whether or not that transaction is legitimate? We want to examine machine learning (ML) models for anti-money laundering (AML) that detect these fraudulent transactions. In such a high-stakes area, it is paramount that humans are able to understand and interpret the predictions from these models via explanations from interpretability tools. However, how should we navigate the trade-offs between certain properties of such explanations? Specifically, what properties of explanations do AML domain experts value the most for AML machine learning models?

To answer these questions, we first train an XGBoost model on the Bank Account Fraud (BAF) dataset, resulting in a TPR of 0.51, an FPR of 0.05, and an AUC of 0.88. Then, we generate initial explanations for our model via SHAP and analyze their fidelity, robustness, compactness, and homogeneity. We then meet with AML domain experts in order to determine what they prioritize throughout the AML process and how that relates to our explanation properties. After both adding regularization to our model and utilizing smoothed SHAP for our explanations, we see a slight decrease in model performance but a drastic improvement to our metrics across the board.
