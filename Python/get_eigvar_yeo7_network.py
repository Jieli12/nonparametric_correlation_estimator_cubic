"""
Author      : Jie Li
Date        : 2026-09-11 22:34:02
Last Edited : 2026-09-12 09:15:17
Last Author : Jie Li
File Path   : /nonparametric_correlation_estimator_cubic/Python/get_eigvar_yeo7_network.py
Description :

Calculate eigenvalue variance and Yeo-7 connectivity network for each band. Save results in separate files for each band.






Copyright (c) 2026, Jie Li, lijie_12@outlook.com
All Rights Reserved.
"""

# %% ========================================================================
# 				import  library
# ===========================================================================
import numpy as np
from statsmodels.stats.correlation_tools import cov_nearest

# %% ========================================================================
# 				check the ROI names
# ===========================================================================
results = np.load("../Results/bandwidths_delta_dfreq_none.npy", allow_pickle=True)
pair_index = results.item()["pairs"]
del results
roi_names = np.load("../Results/roi_names.npy")


# %% define the Yeo7 network names and their short names
YEO7_NAMES = [
    "Visual",
    "Somatomotor",
    "Dorsal Attention",
    "Ventral Attention",
    "Limbic",
    "Frontoparietal",
    "Default",
]


ROI_PATTERNS = {
    "_Vis_": "Visual",
    "_SomMot_": "Somatomotor",
    "_DorsAttn_": "Dorsal Attention",
    "_SalVentAttn_": "Ventral Attention",
    "_Limbic_": "Limbic",
    "_Cont_": "Frontoparietal",
    "_Default_": "Default",
}


# %% ========================================================================
# 			extract the Yeo7 network labels from the ROI names
# ===========================================================================
def extract_yeo7(roi_names: np.ndarray) -> np.ndarray:
    """Infer the Yeo-7 network from standard Schaefer-200 7Networks labels."""
    labels = []
    for roi in roi_names:
        matches = [
            network for pattern, network in ROI_PATTERNS.items() if pattern in roi
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Cannot uniquely identify Yeo-7 network from ROI name: {roi}"
            )
        labels.append(matches[0])

    labels = np.asarray(labels, dtype=str)
    if set(labels) != set(YEO7_NAMES):
        raise ValueError(
            f"Expected all seven Yeo-7 networks; found {sorted(set(labels))}"
        )
    return labels


labels = extract_yeo7(roi_names)


# %% ========================================================================
# 				build edge group
# ===========================================================================
def build_edge_indices(labels: np.ndarray):
    """Return upper-triangle indices and edge groups for all 7x7 pairs."""
    iu, ju = np.triu_indices(200, k=1)
    edge_groups: dict[tuple[int, int], np.ndarray] = {}
    for a in range(7):
        for b in range(a, 7):
            mask = ((labels[iu] == YEO7_NAMES[a]) & (labels[ju] == YEO7_NAMES[b])) | (
                (labels[iu] == YEO7_NAMES[b]) & (labels[ju] == YEO7_NAMES[a])
            )
            edge_groups[(a, b)] = np.flatnonzero(mask)
    return iu, ju, edge_groups


iu, ju, edge_groups = build_edge_indices(labels)


# %% ========================================================================
# 			positive definitiveness check Higham (2002)
# ===========================================================================
def check_positive_definite(matrix):
    """Check positive definiteness by verifying that all eigenvalues are strictly greater than 0."""
    eigenvalues = np.linalg.eigvalsh(matrix)
    min_eig = np.min(eigenvalues)
    # Check if the minimum eigenvalue is greater than a small positive threshold to determine positive definiteness
    is_pd = min_eig > 1e-8
    return is_pd, min_eig


# %% ========================================================================
# 			Main computation
# ===========================================================================
n_rois = len(roi_names)
bands = ["delta", "theta", "alpha", "beta", "slow_gamma", "fast_gamma"]
threshold = 0.5
num_observations = 5740

# Map each ROI to its corresponding Yeo-7 network index (0..6)
roi_to_network_idx = np.array([YEO7_NAMES.index(net) for net in labels])

for ind, band in enumerate(bands):
    print(f"\n[Processing] {band} band...")
    results = np.load(f"../Results/dynamic_corr_{band}.npy")

    # Initialize arrays for this band
    eig_vars = np.zeros(num_observations)
    yeo7_networks = np.zeros((7, 7, num_observations))

    for t in range(num_observations):
        # Build correlation matrix
        corr_matrix_2d = np.zeros((n_rois, n_rois))
        corr_matrix_2d[iu, ju] = results[:, t]
        corr_matrix_2d = corr_matrix_2d + corr_matrix_2d.T + np.eye(n_rois)

        # Check positive definiteness
        is_pd_before, min_eig_before = check_positive_definite(corr_matrix_2d)

        # Apply Higham algorithm if not positive definite
        if not is_pd_before:
            matrix_pd = cov_nearest(corr_matrix_2d, method="clipped", threshold=1e-6)
        else:
            matrix_pd = corr_matrix_2d

        # Calculate eigenvalue variance
        eig_vals = np.linalg.eigvalsh(matrix_pd)
        eig_vars[t] = np.var(eig_vals)

        # Threshold the correlation matrix
        matrix_pd[np.abs(matrix_pd) < threshold] = 0.0

        # Clip to prevent arctanh overflow
        matrix_clipped = np.clip(matrix_pd, -0.9991, 0.9991)
        z_matrix = np.arctanh(matrix_clipped)

        # Compress to 7x7 Yeo-7 network
        network_matrix = np.zeros((7, 7))
        for i in range(7):
            for j in range(7):
                rois_i = np.where(roi_to_network_idx == i)[0]
                rois_j = np.where(roi_to_network_idx == j)[0]

                block = z_matrix[np.ix_(rois_i, rois_j)]
                n_rois_i = len(rois_i)
                n_rois_j = len(rois_j)

                if i == j:
                    # Within-network: build a diagonal mask to exclude self-correlation elements
                    mask_2d = np.ones((n_rois_i, n_rois_j), dtype=bool)
                    np.fill_diagonal(mask_2d, False)
                    if n_rois_i > 1:
                        mean_z = np.mean(block[mask_2d])
                    else:
                        mean_z = 0.0
                else:
                    # mean
                    mean_z = np.mean(block)

                network_matrix[i, j] = mean_z

        # Convert back from Fisher z-score
        network_matrix = np.tanh(network_matrix)
        yeo7_networks[:, :, t] = network_matrix

        # Print progress
        if (t + 1) % 500 == 0:
            print(f"  Progress: {t+1}/{num_observations} time steps completed")

    # Save results for this band
    output_file = f"../Results/eigvar_yeo7_network_{band}.npz"
    np.savez(output_file, eig_vars=eig_vars, yeo7_networks=yeo7_networks, band=band)

    print(f"[Done] Save {band} band results to {output_file}")

    print(f"  - eigenvalue variance shape: {eig_vars.shape}")
    print(f"  - Yeo-7 networks shape: {yeo7_networks.shape}")

print("\n[Done] All bands processed!")
