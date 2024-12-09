# Calculate the metrics for the explainer.
# https://arxiv.org/abs/2211.05667


import numpy as np
import pandas as pd
from explain import explain
from scipy.stats import entropy
from sklearn.metrics import mean_squared_error


def calculate_metrics():
    # Get the testing data, model, and explainer.
    X_test, model, explainer = explain()

    def loss_based_fidelity(E, f, X=X_test):
        """Calculate the loss-based fidelity of the explainer `E` for the model `f` on the dataset `X`."""
        explanations = E(X).values
        reconstructed = explanations.sum(axis=1)
        predictions = f.predict(X)
        return mean_squared_error(predictions, reconstructed)

    def sample_points_l2_ball(x, r, num_samples):
        """Generate random points within an L2 ball of radius `r` around `x`."""
        samples = []
        for _ in range(num_samples):
            # Generate random direction.
            direction = np.random.randn(len(x))
            direction /= np.linalg.norm(direction)

            # Generate random radius within [0, r].
            radius = np.random.uniform(0, r)

            # Compute perturbation and add to x.
            samples.append(x + radius * direction)
        return np.array(samples)

    def local_stability(E, x, r=0.1, num_samples=10000, epsilon=1e-4):
        """Calculate the local stability of the explainer `E` on the instance `x`."""

        # Generate perturbations around x.
        E_x = E([x]).values[0]
        perturbations = sample_points_l2_ball(x, r, num_samples)
        E_perturbations = E(perturbations).values
        assert len(perturbations) == len(E_perturbations)

        # Compute stability ratio for each perturbation.
        max_ratio = 0
        for i in range(len(perturbations)):
            perturbation, E_perturbation = perturbations[i], E_perturbations[i]
            explanation_diff = np.linalg.norm(E_perturbation - E_x)
            perturbation_diff = np.linalg.norm(perturbation - x)
            if perturbation_diff < epsilon:
                continue
            ratio = explanation_diff / perturbation_diff
            max_ratio = max(max_ratio, ratio)

        return max_ratio

    def entropy_complexity(E, X=X_test):
        """Calculate the entropy-complexity of the explainer `E` on the dataset `X`."""
        explanations = E(X).values
        entropy_values = []
        for explanation in explanations:
            probabilities = np.abs(explanation) / np.sum(np.abs(explanation))
            entropy_values.append(entropy(probabilities))
        return np.mean(entropy_values)

    def faithfulness_loss_per_group(E, f, groups, X=X_test):
        """
        Calculate the faithfulness loss per group of the explainer `E` for the model `f` on the dataset `X`.

        The `groups` parameter is a dictionary of groups with indices, e.g., `{"group1": [0, 1, 2], "group2": [3, 4, 5]}`.
        """
        group_losses = {}
        for group, indices in groups.items():
            X_group = X.iloc[indices]
            group_losses[group] = loss_based_fidelity(E, f, X_group)
        return group_losses

    # Calculate the metrics.
    print("Loss-Based Fidelity:", loss_based_fidelity(explainer, model))
    print("Local Stability:", local_stability(explainer, X_test.iloc[21]))
    print("Entropy-Complexity:", entropy_complexity(explainer))

    X_test["income_quartile"] = pd.cut(X_test["income"], bins=4, labels=False)
    groups = X_test.groupby("income_quartile").groups
    X_test.drop(columns=["income_quartile"], inplace=True)

    print(
        "Faithfulness Loss Per Income Group:",
        faithfulness_loss_per_group(explainer, model, groups),
    )

    X_test["customer_age_quartile"] = pd.cut(
        X_test["customer_age"], bins=4, labels=False
    )
    groups = X_test.groupby("customer_age_quartile").groups
    X_test.drop(columns=["customer_age_quartile"], inplace=True)

    print(
        "Faithfulness Loss Per Customer Age Group:",
        faithfulness_loss_per_group(explainer, model, groups),
    )

    housing_status_cols = [
        "housing_status_" + x for x in ("BA", "BB", "BC", "BD", "BE", "BF", "BG")
    ]
    X_test["housing_status"] = np.argmax(X_test[housing_status_cols].values, axis=1)
    groups = X_test.groupby("housing_status").groups
    X_test.drop(columns=["housing_status"], inplace=True)

    print(
        "Faithfulness Loss Per Housing Status Group:",
        faithfulness_loss_per_group(explainer, model, groups),
    )


if __name__ == "__main__":
    calculate_metrics()
