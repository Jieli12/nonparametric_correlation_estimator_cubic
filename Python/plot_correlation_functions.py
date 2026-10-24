"""
Author      : Jie Li
Date        : 2026-08-12 09:44:28
Last Edited : 2026-08-12 09:57:35
Last Author : Jie Li
File Path   : /nonparametric_correlation_estimator/Python/plot_correlation_functions.py
Description :








Copyright (c) 2026, Jie Li, lijie_12@outlook.com
All Rights Reserved.
"""

import matplotlib.pyplot as plt
import numpy as np

# 1. Define the domain u from -1 to 1
u = np.linspace(-1, 1, 500)

# 2. Define the dictionary of functions with LaTeX labels
functions = {
    r"Parabolic": 1 - 2 * u**2,
    r"Sinusoidal": np.sin(np.pi * u / 2),
    r"Cosine": np.cos(np.pi * u / 2),
    r"Cubic": u**3,
    r"Linear": u,
    r"Exponential": (np.exp(u) - np.exp(-1)) / (np.exp(1) - np.exp(-1)),
    r"Semi-circle": np.sqrt(1 - u**2),
    r"Inverse sinusoidal": (2 / np.pi) * np.arcsin(u),
    r"Chebyshev polynomial": 4 * u**3 - 3 * u,
}
# Rescale as in the simulations, rho_c(u) = RHO_SCALE * rho(u) (Assumption (A3))
RHO_SCALE = 0.999999
functions = {label: RHO_SCALE * y for label, y in functions.items()}
# functions = {
#     r"Parabolic: $\rho(u) = 1 - 2u^2$": 1 - 2 * u**2,
#     r"Sinusoidal: $\rho(u) = \sin(\pi u / 2)$": np.sin(np.pi * u / 2),
#     r"Cosine: $\rho(u) = \cos(\pi u / 2)$": np.cos(np.pi * u / 2),
#     r"Cubic: $\rho(u) = u^3$": u**3,
#     r"Linear: $\rho(u) = u$": u,
#     r"Exponential: $\rho(u) = \frac{\exp(u) - \exp(-1)}{\exp(1) - \exp(-1)}$": (
#         np.exp(u) - np.exp(-1)
#     )
#     / (np.exp(1) - np.exp(-1)),
#     r"Upper half semi-circle: $\rho(u) = \sqrt{1 - u^2}$": np.sqrt(1 - u**2),
#     r"Normalised inverse sinusoidal: $\rho(u) = \frac{2}{\pi}\arcsin(u)$": (2 / np.pi)
#     * np.arcsin(u),
#     r"3rd Chebyshev polynomial: $\rho(u) = 4u^3 - 3u$": 4 * u**3 - 3 * u,
# }

# 3. Create a clean, elegant plotting layout
plt.figure(figsize=(10, 6), dpi=800)
plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)

# 4. Use a beautiful discrete colormap (tab10 provides vibrant, modern colors)
colors = plt.cm.tab10(np.linspace(0, 1, len(functions)))

# 5. Plot each function
for idx, (label, y_values) in enumerate(functions.items()):
    plt.plot(u, y_values, label=label, color=colors[idx], linewidth=4)

# 6. Customize axes, grid, and borders
plt.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
plt.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
# plt.title(
#     "Comparison of Mathematical Functions $\\rho(u)$",
#     fontsize=14,
#     fontweight="bold",
#     pad=15,
# )
plt.xlabel("$u$", fontsize=14)
plt.ylabel("$\\rho_{c}(u)$", fontsize=14)
plt.xlim([-1.05, 1.05])
plt.ylim([-1.2, 1.2])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, linestyle=":", alpha=0.6)

# 7. Position the legend nicely on the right side outside the plot box
plt.legend(
    bbox_to_anchor=(1.04, 0.5),
    loc="center left",
    borderaxespad=0,
    fontsize=14,
    frameon=True,
)

# 8. Adjust layout to ensure the legend is not cropped out
plt.tight_layout()
plt.savefig(
    "../Latex/figs/correlation_functions.pdf",
    format="pdf",
    bbox_inches="tight",
    dpi=800,
)
# 9. Show the plot
plt.show()
