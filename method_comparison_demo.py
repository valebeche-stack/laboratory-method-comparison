"""
Educational method-comparison example for laboratory medicine.

The dataset is synthetic and contains no patient data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def bland_altman(method_a, method_b):
    """Return means, differences, bias and 95% limits of agreement."""
    method_a = np.asarray(method_a, dtype=float)
    method_b = np.asarray(method_b, dtype=float)

    means = (method_a + method_b) / 2
    differences = method_b - method_a

    bias = np.mean(differences)
    sd = np.std(differences, ddof=1)
    loa_lower = bias - 1.96 * sd
    loa_upper = bias + 1.96 * sd

    return means, differences, bias, loa_lower, loa_upper


def main():
    df = pd.read_csv("example_data.csv")

    x = df["reference_method"]
    y = df["comparison_method"]

    # Correlation is descriptive only and is not a measure of agreement.
    r, p_value = stats.pearsonr(x, y)

    means, differences, bias, loa_lower, loa_upper = bland_altman(x, y)

    print("Method comparison summary")
    print("-------------------------")
    print(f"n = {len(df)}")
    print(f"Pearson r = {r:.3f} (p = {p_value:.3g})")
    print(f"Mean bias (comparison - reference) = {bias:.3f}")
    print(f"95% limits of agreement = {loa_lower:.3f} to {loa_upper:.3f}")

    # Scatter plot with line of identity
    plt.figure()
    plt.scatter(x, y)
    minimum = min(x.min(), y.min())
    maximum = max(x.max(), y.max())
    plt.plot([minimum, maximum], [minimum, maximum], linestyle="--")
    plt.xlabel("Reference method")
    plt.ylabel("Comparison method")
    plt.title("Method comparison")
    plt.tight_layout()
    plt.show()

    # Bland–Altman plot
    plt.figure()
    plt.scatter(means, differences)
    plt.axhline(bias, linestyle="-")
    plt.axhline(loa_lower, linestyle="--")
    plt.axhline(loa_upper, linestyle="--")
    plt.xlabel("Mean of paired measurements")
    plt.ylabel("Difference (comparison - reference)")
    plt.title("Bland–Altman plot")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
